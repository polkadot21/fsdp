import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# --- PATCHING (Mock CUDA) ---
from mocks import MockEvent, MockStreamManager

torch.cuda.Event = MockEvent
torch.cuda.stream = lambda s: s
# We DO NOT patch reduce_scatter here because we want to see if
# Gloo (CPU backend) crashes on shape mismatch, which confirms the logic bug.
# ----------------------------

from fsdp.buffer_pool import StaticBufferPool  # noqa
from fsdp.config import Setup  # noqa
from fsdp.fsdp_layer import FSDPLayer  # noqa
from fsdp.models.tiny_model import TinyModel  # noqa


def worker(rank, world_size):
    # Re-apply patches
    torch.cuda.Event = MockEvent
    torch.cuda.stream = lambda s: s

    # Use GLOO for CPU
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)

    # --- ASYMMETRIC CONFIG ---
    # Distinct dimensions to debug shape mismatches
    # in_dim (128) != dim (256) != ff_dim (512)
    cfg = Setup(
        model_type="poc",
        in_dim=128,
        dim=256,
        n_heads=4,
        ff_dim=512,
        n_layers=2,  # Keep it simple: 2 layers
        batch=2,
        T=16,
        overlap=False,  # Strict sync mode for debugging
    )

    streams = MockStreamManager(torch.device("cpu"))

    # Create Model
    # We construct TinyModel manually to ensure we control everything
    base_model = TinyModel(cfg.in_dim, cfg.dim, cfg.n_heads, cfg.ff_dim, cfg.n_layers)

    # Buffer Size Calculation
    sizes = [sum(p.numel() for p in blk.parameters()) for blk in base_model.blocks]
    bufpool = StaticBufferPool.from_block_sizes(sizes, torch.device("cpu"), torch.float32)

    # Wrap Blocks
    fsdp_layers = []
    for i, blk in enumerate(base_model.blocks):
        layer = FSDPLayer(blk, i, streams, bufpool, cfg)
        fsdp_layers.append(layer)

        # --- DEBUG: ADD TENSOR HOOKS ---
        # This prints shapes immediately when a gradient is computed for a param
        for name, p in layer.module.named_parameters():

            def get_printer(n, block_i):
                def printer(grad):
                    print(
                        f"[Rank {rank}] Backward: {n} (Block {block_i}) | Grad Shape: {grad.shape}"
                    )
                    return None

                return printer

            # We register on the Parameter wrapper.
            # Note: If p.data is replaced, the hook *should* stick to the wrapper.
            if p.requires_grad:
                p.register_hook(get_printer(name, i))

    # Projection layers (not wrapped for this test, just simple DDP/local)
    inp = base_model.inp
    out = base_model.out

    # Data
    # Batch=2, T=16, Dim=128
    x = torch.randn(2, 16, cfg.in_dim, requires_grad=True)
    y_target = torch.randn(2, 16, cfg.in_dim)

    print(f"[Rank {rank}] Starting Forward...")

    # 1. Forward
    # Prefetch Block 0
    fsdp_layers[0].prefetch_forward()

    h = inp(x)  # 128 -> 256

    for i, layer in enumerate(fsdp_layers):
        if i + 1 < len(fsdp_layers):
            fsdp_layers[i + 1].prefetch_forward()

        # Run Block
        h = layer(h)

    final = out(h)  # 256 -> 128

    print(f"[Rank {rank}] Forward Done. Output: {final.shape}")

    # 2. Backward
    loss = ((final - y_target) ** 2).sum()
    print(f"[Rank {rank}] Starting Backward...")

    # Prime Backward
    fsdp_layers[-1].prefetch_backward()

    loss.backward()

    print(f"[Rank {rank}] SUCCESS.")
    dist.destroy_process_group()


def run_test():
    world_size = 2
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"  # Use different port
    mp.spawn(worker, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    run_test()
