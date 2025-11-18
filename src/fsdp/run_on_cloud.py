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
    print(f"Setting env for rank: {rank}")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    msg = f"Env: {os.environ['RANK']}, local rank: {os.environ['LOCAL_RANK']}"
    print(msg)
    print("################################################")

    print(
        f"[worker {rank}] torch.cuda.is_available={torch.cuda.is_available()}, "
        f"device_count={torch.cuda.device_count()}"
    )
    ddp_init(rank, world_size)
    try:
        train_one_rank(cfg.cloud, logdir=cfg.logs.dir, profile_steps=cfg.profiler.n_steps)
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

    world_size = torch.cuda.device_count()
    if world_size < 2:
        err_msg = f"For FSDP two or more GPUs required, current count: {world_size}"
        print(err_msg)
        raise exceptions.NotEnoughGpuError(err_msg)

    print(f"Launching experiment with {world_size} GPUs")

    cfg = get_cfg()
    # ------------------------------------------------------------------
    # Multi-GPU case
    # ------------------------------------------------------------------
    print("Spawning distributed workers...")
    mp.spawn(
        _worker,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True,
    )
