import os

import torch
import torch.multiprocessing as mp
from loguru import logger

from fsdp.config import ModelType, get_cfg, get_model_config
from fsdp.train_loop import train_worker


def run_on_cloud(mode: str | ModelType = "poc"):
    """
    Main entry point for running FSDP experiments.
    Can be called from CLI or Jupyter Notebook.

    Args:
        mode: "poc" (fast, overlap check) or "giant" (compute heavy).
              Defaults to "poc" for safer quick testing.
    """
    # 1. Handle Mode Selection (String -> Enum)
    if isinstance(mode, str):
        try:
            # Normalize string input (e.g., "POC" -> ModelType.POC)
            mode = ModelType(mode.lower())
        except ValueError:
            valid_modes = [m.value for m in ModelType]
            logger.error(f"Invalid mode: '{mode}'. Valid options: {valid_modes}")
            return

    # 2. Setup Environment & Config
    # get_cfg() has a side-effect: it sets MASTER_ADDR/PORT/WORLD_SIZE
    # if they are missing (crucial for Jupyter).
    base_cfg = get_cfg()

    # Robust World Size Detection
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Fallback for Jupyter if get_cfg didn't set it (safety net)
        world_size = torch.cuda.device_count()

    if world_size < 2:
        logger.error(f"FSDP requires at least 2 GPUs. Found: {world_size}")
        logger.warning("If you are testing on CPU, run 'make test' instead.")
        return

    logger.info(f"Launching FSDP Experiment on {world_size} GPUs. Mode: {mode.value.upper()}")

    # 3. Prepare Config
    cfg = base_cfg.model_copy()
    cfg.train = get_model_config(mode)
    cfg.logs_dir = f"logs/{mode.value}"
    os.makedirs(cfg.logs_dir, exist_ok=True)

    # 4. Spawn Workers
    # nprocs=world_size spawns one process per GPU
    mp.spawn(train_worker, args=(world_size, cfg), nprocs=world_size)
