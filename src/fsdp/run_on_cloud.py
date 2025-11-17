import os

import torch
import torch.multiprocessing as mp

from fsdp import exceptions
from fsdp.config import Config, get_cfg
from fsdp.dist_utils import ddp_cleanup, ddp_init
from fsdp.train_loop import train_one_rank


def _worker(rank: int, world_size: int, cfg: Config) -> None:
    """
    Child worker. Must set rank-specific env vars **before** ddp_init().
    """

    # ---- Required for torch.distributed.init_process_group(init_method="env://") ----
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # For debugging
    print(
        f"[worker {rank}] torch.cuda.is_available={torch.cuda.is_available()}, "
        f"device_count={torch.cuda.device_count()}"
    )

    # Let ddp_init read these correctly
    use_cuda, backend = ddp_init(rank, world_size)

    if use_cuda:
        print(
            f"[worker {rank}] current_device={torch.cuda.current_device()}, "
            f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
        )

    print(f"[worker {rank}] CUDA={use_cuda} backend={backend}")

    try:
        train_one_rank(cfg.cloud, logdir=cfg.logs.dir, profile_steps=8)
    finally:
        ddp_cleanup()


def run_on_cloud() -> None:
    """
    Jupyter-friendly:
    - Auto-select world_size based on available GPUs
    - Uses mp.spawn for multi-GPU.
    """

    if not torch.cuda.is_available():
        err_msg = f"CUDA available: {torch.cuda.is_available()}"
        print(err_msg)
        raise exceptions.CudaNotFoundError(err_msg)

    cfg = get_cfg()
    world_size = torch.cuda.device_count()
    print(f"Launching with world_size={world_size} with {world_size} GPUs")

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
            args=(world_size, cfg.cloud),
            nprocs=world_size,
            join=True,
        )

    # ------------------------------------------------------------------
    # Single-process fallback (CPU or 1 GPU)
    # ------------------------------------------------------------------
    else:
        _worker(rank=0, world_size=1, cfg=cfg)
