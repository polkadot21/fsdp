import torch


class TwoBufferPool:
    """
    YaFSDP-style two global full-parameter buffers:
      - buf_even used by even-indexed FSDP blocks (0,2,...)
      - buf_odd  used by odd-indexed  FSDP blocks (1,3,...)
    Each buffer size = max full flat size across all blocks (floats).
    We only ever have at most one EVEN and one ODD layer active concurrently,
    so two buffers suffice and bound peak memory.
    """

    def __init__(self, max_full_numel: int, device: torch.device, dtype=torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self.maxn = max_full_numel
        if self.maxn <= 0:
            raise ValueError("max_full_numel must be > 0")

        self.buf_even = torch.empty(self.maxn, device=self.device, dtype=self.dtype)
        self.buf_odd = torch.empty(self.maxn, device=self.device, dtype=self.dtype)

    def buffer_for(self, block_idx: int) -> torch.Tensor:
        return self.buf_even if (block_idx % 2 == 0) else self.buf_odd
