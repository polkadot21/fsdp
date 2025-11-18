import functools
import os

import torch
from pydantic import Field
from pydantic_settings import BaseSettings


class BaseSetup(BaseSettings):
    """
    Common fields for any training configuration.
    """

    in_dim: int = Field(..., description="Input embedding dimension")
    dim: int = Field(..., description="Model hidden size")
    n_heads: int = Field(..., description="Number of attention heads")
    ff_dim: int = Field(..., description="Feed-forward inner dimension")
    n_layers: int = Field(..., description="Number of Transformer blocks")
    batch: int = Field(..., description="Batch size per rank")
    T: int = Field(..., description="Sequence length")
    steps: int = Field(..., description="Steps per validation epoch")
    lr: float = Field(..., description="Learning rate")
    wd: float = Field(..., description="Weight decay")

    sync_collectives: bool = Field(False, description="Disable overlap: run AG/RS synchronously")

    class Config:
        extra = "ignore"


class CPUSetup(BaseSetup):
    """
    Small config for CPU or single-GPU debugging.
    """

    in_dim: int = 128
    dim: int = 256
    n_heads: int = 8
    ff_dim: int = 1024
    n_layers: int = 4
    batch: int = 4
    T: int = 64
    steps: int = 50
    lr: float = 1e-3
    wd: float = 0.0


class CloudSetup(BaseSetup):
    """
    'Fat' configuration which makes communication heavy enough
    to expose compute/comm overlap on multi-GPU A100/H100.
    """

    in_dim: int = 2048
    dim: int = 4096
    n_heads: int = 32
    ff_dim: int = 16384
    n_layers: int = 8
    batch: int = 4
    T: int = 256
    steps: int = 100
    lr: float = 1e-3
    wd: float = 0.0


class Logs(BaseSettings):
    dir: str = Field("logs")


class Profiler(BaseSettings):
    n_steps: int = Field(8)


# -----------------------------
#       Unified Config
# -----------------------------
class Config(BaseSettings):
    """
    Top-level unified config object.
    Access like cfg.cpu or cfg.cloud.
    """

    cpu: CPUSetup = CPUSetup()
    cloud: CloudSetup = CloudSetup()
    logs: Logs = Logs()
    profiler: Profiler = Profiler()

    class Config:
        extra = "ignore"


@functools.lru_cache
def get_cfg() -> Config:
    """
    Cached singleton config instance.
    Imported everywhere to avoid reallocation.
    """

    world_size = torch.cuda.device_count()

    # ------------------------------------------------------------------
    # For mp.spawn + env://, we MUST set MASTER_ADDR/MASTER_PORT manually.
    # torchrun usually does this, but Jupyter notebook does NOT.
    # ------------------------------------------------------------------
    print(f"Setting env for torch multiprocessing for world_size: {world_size}")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    print(f"Env addr: {os.environ['MASTER_ADDR']}, port: {os.environ['MASTER_PORT']}")
    print("################################################")

    return Config()
