import torch
import torch.multiprocessing as mp

from fsdp.data import cloud_cfg, cpu_cfg
from fsdp.dist_utils import ddp_cleanup, ddp_init
from fsdp.train_loop import train_one_rank


def _worker(rank, world_size, cfg, logdir):
    use_cuda, backend = ddp_init(rank, world_size)
    print(f"Worker: {rank} running with CUDA: {use_cuda} and DDP backend: {backend}")
    try:
        train_one_rank(cfg, logdir=logdir, profile_steps=8)
    finally:
        ddp_cleanup()


def run_on_cloud(world_size=None, fat=True, logdir="logs"):
    """
    Jupyter-friendly entry point.
    - If CUDA available and multiple GPUs present, spawns one process per GPU.
    - Otherwise falls back to single-process CPU run (no NCCL needed).
    """
    print(f"=== Running with world size: {world_size} =====")
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    cfg = cloud_cfg() if fat else cpu_cfg()
    msg = f"Launching with world_size={world_size} (CUDA={torch.cuda.is_available()})"
    print(msg)
    if world_size > 1:
        mp.spawn(_worker, nprocs=world_size, args=(world_size, cfg, logdir), join=True)
    else:
        _worker(rank=0, world_size=1, cfg=cfg, logdir=logdir)
