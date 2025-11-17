import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from fsdp import config, consts
from fsdp.buffer_pool import TwoBufferPool
from fsdp.data import make_batch
from fsdp.dist_utils import barrier, world_info
from fsdp.dummy_fsdp import DIYFSDPBlockAB
from fsdp.model import TinyModel
from fsdp.profiling_utils import analyze_profiler, print_ascii_gantt

_PRINTED_STEP_ONCE_DEVICE: bool = False


class FSDPWrappedModel(torch.nn.Module):
    def __init__(self, cfg: config.BaseSetup, device: torch.device, lr, wd):
        super().__init__()
        self.cfg = cfg
        self.dev = device
        print(f"[FSDPWrappedModel] constructing on device={device}")

        m = TinyModel(
            cfg.in_dim,
            cfg.dim,
            cfg.n_heads,
            cfg.ff_dim,
            cfg.n_layers,
        ).to(device)

        print(f"[FSDPWrappedModel] TinyModel.inp.weight.device={m.inp.weight.device}")

        dummy_blocks = []
        sizes = []
        for i, blk in enumerate(m.blocks):
            wrap = DIYFSDPBlockAB(device, blk, block_idx=i, bufpool=None, lr=lr, wd=wd)
            sizes.append(wrap.shard_size * (dist.get_world_size() if dist.is_initialized() else 1))
            dummy_blocks.append(wrap)

        max_full = max(sizes)
        bufpool = TwoBufferPool(max_full, device=torch.device(device), dtype=torch.float32)

        self.inp = m.inp
        self.blocks = torch.nn.ModuleList(
            [
                DIYFSDPBlockAB(device, blk.mod, block_idx=i, bufpool=bufpool, lr=lr, wd=wd)
                for i, blk in enumerate(dummy_blocks)
            ]
        )
        print(
            f"[FSDPWrappedModel] real block 0 first_param_device="
            f"{next(self.blocks[0].mod.parameters()).device}"
        )

        self.out = m.out

    def forward(self, x: torch.Tensor):
        # schedule: prefetch block0, then loop blocks
        self.blocks[0].prefetch_params_async()
        self.blocks[0].materialize_params()
        x = self.inp(x)
        for i, blk in enumerate(self.blocks):
            # prefetch next while computing current
            if i + 1 < len(self.blocks):
                self.blocks[i + 1].prefetch_params_async()
            with record_function(f"FWD/block_{i}"):
                x = blk(x)
            if i + 1 < len(self.blocks):
                self.blocks[i + 1].materialize_params()
        return self.out(x)

    @torch.no_grad()
    def step_all(self):
        ok = True
        for blk in reversed(self.blocks):
            ok &= blk.step_if_ready()
        return ok


def warmup(cfg: config.BaseSetup, dev: torch.device, model, n_steps: int = 3):
    print(f"[warmup] running on device={dev}")
    for step in range(n_steps):
        x, y = make_batch(cfg.batch, cfg.T, cfg.in_dim, dev)
        out = model(x)
        if step == 0:
            print(f"[warmup] x.device={x.device}, y.device={y.device}, out.device={out.device}")
        loss = F.mse_loss(out, y)
        loss.backward()
        model.step_all()
        if dev.type == consts.Device.CUDA:
            torch.cuda.synchronize()
    print(f"===== Warmup complete with {n_steps} steps")


def step_once(cfg: config.BaseSetup, dev: torch.device, model) -> float:
    global _PRINTED_STEP_ONCE_DEVICE

    x, y = make_batch(cfg.batch, cfg.T, cfg.in_dim, dev)
    t0 = time.time()
    out = model(x)

    if not _PRINTED_STEP_ONCE_DEVICE:
        print(f"[step_once] x.device={x.device}, y.device={y.device}, out.device={out.device}")
        _PRINTED_STEP_ONCE_DEVICE = True

    loss = F.mse_loss(out, y)
    loss.backward()
    model.step_all()
    if dev.type == consts.Device.CUDA:
        torch.cuda.synchronize()
    return time.time() - t0


def train_one_rank(
    data_cfg: config.BaseSetup,
    logdir: str = "logs",
    profile_steps: int = 8,
) -> None:
    rank, world = world_info()
    dev = (
        torch.device(consts.Device.CUDA, rank)
        if torch.cuda.is_available()
        else torch.device(consts.Device.CPU)
    )
    print(
        f"[train_one_rank] rank={rank} world={world} dev={dev} "
        f"cuda_available={torch.cuda.is_available()}"
    )

    model = FSDPWrappedModel(data_cfg, device=dev, lr=data_cfg.lr, wd=data_cfg.wd).to(dev)
    print(f"[train_one_rank] rank={rank} first_param_device=" f"{next(model.parameters()).device}")

    warmup(data_cfg, dev, model)
    # Simple runtime printout
    t = sum(step_once(data_cfg, dev, model) for _ in range(10)) / 10
    if rank == 0:
        print(f"[rank{rank}] avg step: {t*1e3:.1f} ms  (world={world})")

    print("Starting profiling")
    os.makedirs(logdir, exist_ok=True)
    trace_path = os.path.join(logdir, f"trace_rank{rank}.json")

    # All ranks enter profiling at the same time
    barrier()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        # ensure profiler starts after all ranks enter it
        barrier()

        for i in range(profile_steps):
            with record_function(f"TRAIN_STEP/rank{rank}/step{i}"):
                step_once(data_cfg, dev, model)

        # ensure everything is completed before profiler closes
        if dev.type == consts.Device.CUDA:
            torch.cuda.synchronize()

        barrier()

    prof.export_chrome_trace(trace_path)
    analyze_profiler(prof, rank)
    print_ascii_gantt(prof, rank)

    if rank == 0:
        print(f"Saved trace: {trace_path}")

    print("Short training loop to verify learning")
    for epoch in range(1, 3):
        losses = []
        for _ in range(data_cfg.steps):
            x, y = make_batch(data_cfg.batch, data_cfg.T, data_cfg.in_dim, dev)
            out = model(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            model.step_all()
            if dev.type == consts.Device.CUDA:
                torch.cuda.synchronize()

            losses.append(loss.item())
        m = sum(losses) / len(losses)
        print(f"[rank{rank}] epoch {epoch} loss={m:.4f}")
        barrier()
