import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# Import MockEvent explicitly
from mocks import MockEvent, MockStreamManager

# --- CRITICAL: Patch BEFORE other imports that might use torch.cuda ---
torch.cuda.Event = MockEvent
torch.cuda.stream = lambda s: s
# --------------------------------------------------------------------

from fsdp.buffer_pool import StaticBufferPool  # noqa
from fsdp.config import Setup  # noqa
from fsdp.fsdp_layer import FSDPLayer  # noqa


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(10, 10))
        self.b = nn.Parameter(torch.randn(10))

    def forward(self, x):
        return x @ self.w + self.b


def worker(rank, world_size):
    # Re-apply patch inside worker just in case
    torch.cuda.Event = MockEvent
    torch.cuda.stream = lambda s: s

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)

    cfg = Setup(overlap=True)
    streams = MockStreamManager(torch.device("cpu"))
    # Ensure buffer pool is large enough and on CPU
    bufpool = StaticBufferPool(512, torch.device("cpu"), torch.float32)

    model = nn.Sequential(
        TinyLinear(),
        TinyLinear(),
        TinyLinear(),
    )

    fsdp_layers = []
    for i, layer in enumerate(model):
        fsdp_layers.append(FSDPLayer(layer, i, streams, bufpool, cfg))

    x = torch.randn(2, 10, requires_grad=True)

    print(f"[Rank {rank}] Running Forward...")
    for i, layer in enumerate(fsdp_layers):
        layer.prefetch_forward()

        # DEBUG: Check if materialize works immediately
        layer._materialize()
        if layer.module.b.numel() == 0:
            raise RuntimeError(f"Rank {rank}: Layer {i} bias is still empty after materialize!")

        x = layer(x)

    loss = x.sum()

    print(f"[Rank {rank}] Running Backward...")
    loss.backward()

    print(f"[Rank {rank}] SUCCESS.")
    dist.destroy_process_group()


def test_cpu_fsdp():
    world_size = 2
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(worker, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    test_cpu_fsdp()
