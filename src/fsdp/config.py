import functools
import os
import sys
from enum import Enum

import torch
from loguru import logger
from pydantic_settings import BaseSettings


class ModelType(str, Enum):
    POC = "poc"  # Fast run, visible overlap
    GIANT = "giant"  # Massive compute, overlap is critical


class Setup(BaseSettings):
    model_type: ModelType = ModelType.POC

    # Architecture
    in_dim: int = 4096
    dim: int = 4096
    n_heads: int = 32
    ff_dim: int = 11008
    n_layers: int = 12

    # Training
    batch: int = 4
    T: int = 1024
    steps: int = 10
    lr: float = 1e-4

    # FSDP Flags
    overlap: bool = True


class Profiler(BaseSettings):
    n_steps: int = 4


class Config(BaseSettings):
    train: Setup = Setup()
    logs_dir: str = "logs"
    profiler: Profiler = Profiler()

    class Config:
        extra = "ignore"


def get_model_config(mode: ModelType) -> Setup:
    """
    Factory to produce consistent model configs.
    """
    if mode == ModelType.POC:
        # Balanced Compute/Comm to verify overlap mechanics
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
        # Heavy Compute dominance. Simulates 70B+ layer widths.
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
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    return cfg
