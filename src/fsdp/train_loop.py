import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from fsdp import consts
from fsdp.buffer_pool import TwoBufferPool
from fsdp.data import make_batch
from fsdp.dist_utils import barrier, world_info
from fsdp.dummy_fsdp import DIYFSDPBlockAB
from fsdp.model import TinyModel


class FSDPWrappedModel(torch.nn.Module):
    def __init__(self, cfg, device, lr, wd):
        super().__init__()
        self.cfg = cfg
        self.dev = device
        m = TinyModel(cfg["in_dim"], cfg["dim"], cfg["n_heads"], cfg["ff_dim"], cfg["n_layers"]).to(
            device
        )

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
        self.out = m.out

    def forward(self, x):
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


def warmup(cfg, dev, model, n_steps: int = 3):
    for _ in range(n_steps):
        x, y = make_batch(cfg["batch"], cfg["T"], cfg["in_dim"], dev)
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        model.step_all()
        torch.cuda.synchronize() if dev.type == consts.Device.CUDA else None

    print(f"===== Warmup complete with {n_steps} steps")


def step_once(cfg, dev, model):
    x, y = make_batch(cfg["batch"], cfg["T"], cfg["in_dim"], dev)
    t0 = time.time()
    out = model(x)
    loss = F.mse_loss(out, y)
    loss.backward()
    model.step_all()
    if dev.type == consts.Device.CUDA:
        torch.cuda.synchronize()
    return time.time() - t0


def train_one_rank(cfg, logdir="logs", profile_steps=8):
    rank, world = world_info()
    dev = (
        torch.device(consts.Device.CUDA, rank)
        if torch.cuda.is_available()
        else torch.device(consts.Device.CPU)
    )

    model = FSDPWrappedModel(cfg, device=dev, lr=cfg["lr"], wd=cfg["wd"]).to(dev)

    warmup(cfg, dev, model)
    # Simple runtime printout
    t = sum(step_once(cfg, dev, model) for _ in range(10)) / 10
    if rank == 0:
        print(f"[rank{rank}] avg step: {t*1e3:.1f} ms  (world={world})")

    # Profiler trace per rank
    print("Starting profiling")
    os.makedirs(logdir, exist_ok=True)
    trace_path = os.path.join(logdir, f"trace_rank{rank}.json")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if dev.type == consts.Device.CUDA
        else [ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(profile_steps):
            with record_function(f"TRAIN_STEP/rank{rank}"):
                step_once(cfg, dev, model)

    prof.export_chrome_trace(trace_path)
    if rank == 0:
        print(f"Saved trace: {trace_path}")

    print("Short training loop to verify learning")
    for epoch in range(1, 3):
        losses = []
        for _ in range(cfg["steps"]):
            x, y = make_batch(cfg["batch"], cfg["T"], cfg["in_dim"], dev)
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
