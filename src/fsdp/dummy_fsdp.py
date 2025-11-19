import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from fsdp import config
from fsdp.buffer_pool import TwoBufferPool
from fsdp.dist_utils import world_info


class DIYFSDPBlockAB(nn.Module):
    """
    Per-block FSDP with global two-buffer scheme + event-chained overlap.
    - At rest: only keep flat shard (FP32) + optimizer states on each rank.
    - Forward: all_gather into global parity buffer on gather_stream; compute waits via event.
               params are views into that buffer (no extra alloc). Keep views through backward.
    - Backward: flatten grads; launch reduce_scatter on rs_stream (async); free views after flatten.
    - Step: wait RS; set shard.grad; optimizer.step() on the shard.
    """

    def __init__(
        self,
        cfg: config.BaseSetup,
        device: torch.device,
        module: nn.Module,
        block_idx: int,
        bufpool: TwoBufferPool | None,
        lr=1e-3,
        wd=0.0,
        dtype_full=torch.float32,
        *,
        register_backward_hook: bool,
    ) -> None:
        super().__init__()
        self.mod = module
        self.block_idx = block_idx
        self.bufpool = bufpool
        self.dtype_full = dtype_full

        self.rank, self.world = world_info()
        self.dev = device
        self.mod.to(self.dev)
        logger.debug(
            f"[DIYFSDPBlockAB] rank={self.rank} block_idx={block_idx} "
            f"mod_first_param_device={next(self.mod.parameters()).device}"
        )

        named = [(n, p) for n, p in self.mod.named_parameters(recurse=True) if p.requires_grad]
        named.sort(key=lambda x: x[0])
        self.params = [p for _, p in named]
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        tot = sum(self.numels)
        pad = ((tot + self.world - 1) // self.world) * self.world - tot
        self.pad = pad
        self.shard_size = (tot + pad) // self.world

        if self.rank == 0:
            flat = torch.cat([p.detach().reshape(-1) for p in self.params], 0)
            if pad:
                flat = torch.cat([flat, torch.zeros(pad, device=self.dev, dtype=flat.dtype)], 0)
            chunks = list(flat.chunk(self.world, 0))
        else:
            chunks = None

        shard = torch.empty(self.shard_size, device=self.dev, dtype=self.params[0].dtype)
        dist.scatter(shard, scatter_list=chunks, src=0)

        self.shard = nn.Parameter(shard.to(torch.float32), requires_grad=True)
        self.opt = torch.optim.AdamW([self.shard], lr=lr, weight_decay=wd)

        for p in self.params:
            with torch.no_grad():
                p.data = torch.empty_like(p, device=self.dev)

        # async state
        self._ag_work = None
        self._prefetch_target = None
        self._views_full = None

        self._grad_buf = None
        self._grad_shard = None
        self._rs_work = None

        # Hook to start RS asap
        if register_backward_hook:
            self.mod.register_full_backward_hook(self._post_backward_hook)

        self.sync_collectives = cfg.sync_collectives

    # ---------- collectives ----------
    @torch.no_grad()
    def _all_gather_into(self, out_flat: torch.Tensor, *, async_op: bool):
        chunks = [
            out_flat.narrow(0, i * self.shard_size, self.shard_size).view_as(self.shard)
            for i in range(self.world)
        ]
        return dist.all_gather(chunks, self.shard, async_op=async_op)

    @torch.no_grad()
    def prefetch_params_async(self):
        flat_full = self.bufpool.buffer_for(self.block_idx)
        self._prefetch_target = flat_full

        if self.sync_collectives:
            # SYNC: blocking all-gather
            dist.all_gather_into_tensor(flat_full, self.shard)
            self._ag_work = None
        else:
            # ASYNC: launch and remember Work handle
            self._ag_work = dist.all_gather_into_tensor(flat_full, self.shard, async_op=True)

    @torch.no_grad()
    def materialize_params(self):
        flat_full = self._prefetch_target
        if flat_full is None:
            return

        # For ASYNC mode, ensure AG is done before using buffer
        if self._ag_work is not None:
            self._ag_work.wait()
            self._ag_work = None

        # map parameter views
        if self.pad:
            usable = flat_full[: -self.pad]
        else:
            usable = flat_full

        off = 0
        for p, n in zip(self.params, self.numels, strict=False):
            v = usable.narrow(0, off, n).view(p.shape)
            p.data = v.to(self.dtype_full)
            off += n

        self._views_full = flat_full

    @torch.no_grad()
    def _post_backward_hook(self, module, gin, gout):
        flats = []
        for p in self.params:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            flats.append(g.reshape(-1))
        g_full = torch.cat(flats, 0)
        if self.pad:
            g_full = torch.cat(
                [g_full, torch.zeros(self.pad, device=self.dev, dtype=g_full.dtype)], 0
            )

        self._grad_buf = g_full
        self._views_full = None
        self._prefetch_target = None

        self._grad_shard = torch.empty_like(self.shard, dtype=g_full.dtype)

        if self.sync_collectives:
            # SYNC: blocking reduce-scatter
            dist.reduce_scatter_tensor(
                self._grad_shard,
                self._grad_buf,
                op=dist.ReduceOp.SUM,
            )
            self._rs_work = None
        else:
            # ASYNC: overlap reduce-scatter; wait in step_if_ready
            self._rs_work = dist.reduce_scatter_tensor(
                self._grad_shard,
                self._grad_buf,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

    @torch.no_grad()
    def step_if_ready(self) -> bool:
        if self._grad_buf is None:
            return False

        if self._rs_work is not None:
            self._rs_work.wait()

        assert self._grad_shard is not None
        g32 = self._grad_shard.to(torch.float32) / (self.world if self.world > 1 else 1.0)
        self.shard.grad = g32
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        for p in self.params:
            p.grad = None

        self._grad_buf = None
        self._grad_shard = None
        self._rs_work = None
        return True

    def forward(self, *args, **kwargs):
        return self.mod(*args, **kwargs)
