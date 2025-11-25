import os

import torch.multiprocessing as mp
from loguru import logger

from fsdp.config import ModelType, get_cfg, get_model_config
from fsdp.train_loop import train_worker


def main():
    base_cfg = get_cfg()
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size < 2:
        logger.error("FSDP requires at least 2 GPUs.")
        return

    # Choose mode here.
    # For the user demo, we run the GIANT mode to prove the point.
    mode = ModelType.GIANT

    logger.info(f"Launching FSDP Experiment. Mode: {mode}")

    cfg = base_cfg.model_copy()
    cfg.train = get_model_config(mode)
    cfg.logs_dir = f"logs/{mode.value}"
    os.makedirs(cfg.logs_dir, exist_ok=True)

    mp.spawn(train_worker, args=(world_size, cfg), nprocs=world_size)


if __name__ == "__main__":
    main()
