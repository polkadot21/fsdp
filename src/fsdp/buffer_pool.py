import math

import torch


class StaticBufferPool:
    """
    Two fixed buffers: [Even, Odd].
    Reused cyclically for Forward (0,1,0,1...) and Backward (...,1,0,1,0).
    """

    def __init__(self, buffer_size: int, device: torch.device, dtype: torch.dtype):
        self.buffer_size = buffer_size
        self.device = device

        # The physical memory
        self.buffers = [
            torch.zeros(buffer_size, device=device, dtype=dtype),
            torch.zeros(buffer_size, device=device, dtype=dtype),
        ]

        # Events for synchronization
        # "ready_event": Recorded by Comm when data arrives. Waited on by Compute.
        self.ready_events = [torch.cuda.Event(), torch.cuda.Event()]

        # "free_event": Recorded by Compute when done using buffer. Waited on by Comm.
        # Initially recorded because buffers start empty.
        self.free_events = [torch.cuda.Event(), torch.cuda.Event()]
        for e in self.free_events:
            e.record()

    @classmethod
    def from_block_sizes(cls, sizes: list[int], device: torch.device, dtype: torch.dtype):
        # Find the largest block to size the buffers
        max_size = max(sizes)
        # 128-element alignment for NCCL efficiency
        aligned_size = math.ceil(max_size / 128) * 128
        return cls(aligned_size, device, dtype)

    def get_buffer(self, idx: int) -> torch.Tensor:
        return self.buffers[idx % 2]

    def get_ready_event(self, idx: int) -> torch.cuda.Event:
        return self.ready_events[idx % 2]

    def get_free_event(self, idx: int) -> torch.cuda.Event:
        return self.free_events[idx % 2]
