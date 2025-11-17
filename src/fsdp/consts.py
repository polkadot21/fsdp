from enum import Enum


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class DdpBackend(str, Enum):
    NCCL = "nccl"
    GLOO = "gloo"
