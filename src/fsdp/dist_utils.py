import torch
import torch.distributed as dist

from fsdp import consts


def ddp_init(rank: int, world_size: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(backend=consts.DdpBackend.NCCL, rank=rank, world_size=world_size)

    print(
        f"[ddp_init] rank={rank} set_device({rank}), "
        f"current_device={torch.cuda.current_device()}"
    )

    return


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
