import functools
import os
import sys
from enum import Enum

import torch
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings


class ModelType(str, Enum):
    POC = "poc"
    GIANT = "giant"


class Setup(BaseSettings):
    model_type: ModelType = ModelType.POC
    in_dim: int = 4096
    dim: int = 4096
    n_heads: int = 32
    ff_dim: int = 11008
    n_layers: int = 12
    batch: int = 4
    T: int = 1024
    steps: int = 10
    lr: float = 1e-4
    overlap: bool = True


class Profiler(BaseSettings):
    n_steps: int = 4


class Config(BaseSettings):
    train: Setup = Setup()
    logs_dir: str = "logs"
    # Added field to control logging level
    log_level: str = Field("DEBUG", description="Logging level (DEBUG, INFO, WARNING)")
    profiler: Profiler = Profiler()

    class Config:
        extra = "ignore"


def get_model_config(mode: ModelType) -> Setup:
    """Factory to generate correct model sizes"""
    if mode == ModelType.POC:
        return Setup(
            model_type=ModelType.POC,
            in_dim=2048,
            dim=2048,
            n_heads=16,
            ff_dim=8192,
            n_layers=8,
            batch=4,
            T=512,
            overlap=True,
        )
    elif mode == ModelType.GIANT:
        return Setup(
            model_type=ModelType.GIANT,
            in_dim=8192,
            dim=8192,
            n_heads=64,
            ff_dim=28672,
            n_layers=16,
            batch=1,
            T=1024,
            overlap=True,
        )
    return Setup()


@functools.lru_cache
def get_cfg() -> Config:
    if "RANK" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        os.environ["RANK"] = "0"

    cfg = Config()

    # --- LOGGING SETUP ---
    logger.remove()

    # 1. Console Handler (Stderr)
    #    Uses cfg.log_level (DEBUG) so you see everything in the terminal too.
    logger.add(
        sys.stderr,
        level=cfg.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",  # noqa
    )

    # 2. File Handler (Always DEBUG for post-mortem analysis)
    rank = int(os.environ["RANK"])
    log_file = f"{cfg.logs_dir}/rank_{rank}.log"

    logger.add(
        log_file,
        level="DEBUG",
        rotation="100 MB",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{line} | {message}",
    )

    if rank == 0:
        logger.info(f"Logging configured. Level: {cfg.log_level}. Logfile: {log_file}")

    return cfg
