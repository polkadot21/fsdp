import os

import torch
import torch.multiprocessing as mp

from fsdp.data import cloud_cfg, cpu_cfg
from fsdp.dist_utils import ddp_cleanup, ddp_init
from fsdp.train_loop import train_one_rank


def _worker(rank, world_size, cfg, logdir):
    """
    Child worker. Must set rank-specific env vars **before** ddp_init().
    """

    # ---- Required for torch.distributed.init_process_group(init_method="env://") ----
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Let ddp_init read these correctly
    use_cuda, backend = ddp_init(rank, world_size)

    print(f"[worker {rank}] CUDA={use_cuda} backend={backend}")

    try:
        train_one_rank(cfg, logdir=logdir, profile_steps=8)
    finally:
        ddp_cleanup()


def run_on_cloud(world_size=None, fat=True, logdir="logs"):
    """
    Jupyter-friendly:
    - Auto-select world_size based on available GPUs
    - **Manually sets MASTER_ADDR / MASTER_PORT**, because torchrun
      is NOT launching this notebook.
    - Uses mp.spawn for multi-GPU.
    """

    print(f"=== Running with world size: {world_size} =====")

    # Decide runtime config (fat transformer layers vs small CPU config)
    cfg = cloud_cfg() if fat else cpu_cfg()

    # Detect world size if user didn't set it
    if world_size is None:
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
        else:
            world_size = 1

    print(f"Launching with world_size={world_size} (CUDA={torch.cuda.is_available()})")

    # ------------------------------------------------------------------
    # For mp.spawn + env://, we MUST set MASTER_ADDR/MASTER_PORT manually.
    # torchrun usually does this, but Jupyter notebook does NOT.
    # ------------------------------------------------------------------
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", str(world_size))

    # ------------------------------------------------------------------
    # Multi-GPU case
    # ------------------------------------------------------------------
    if world_size > 1:
        print("Spawning distributed workers...")
        mp.spawn(
            _worker,
            args=(world_size, cfg, logdir),
            nprocs=world_size,
            join=True,
        )

    # ------------------------------------------------------------------
    # Single-process fallback (CPU or 1 GPU)
    # ------------------------------------------------------------------
    else:
        _worker(rank=0, world_size=1, cfg=cfg, logdir=logdir)
