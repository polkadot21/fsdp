import torch
import torch.distributed as dist

from fsdp import consts


def ddp_init(rank: int, world_size: int) -> tuple[bool, consts.DdpBackend]:
    use_cuda = torch.cuda.is_available()
    backend = consts.DdpBackend.NCCL if use_cuda and world_size > 1 else consts.DdpBackend.GLOO
    if world_size > 1:
        if use_cuda:
            torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return use_cuda, backend


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def world_info() -> tuple[int, int]:
    ws = dist.get_world_size() if dist.is_initialized() else 1
    rk = dist.get_rank() if dist.is_initialized() else 0
    return rk, ws


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()
