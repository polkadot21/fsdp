from collections.abc import Sequence

import torch


class TwoBufferPool:
    """
    Two *physical* buffers, but each block gets its own non-overlapping slice.

      - All even-indexed blocks (0,2,4,...) live inside buf_even
      - All odd-indexed  blocks (1,3,5,...) live inside buf_odd

    Each block's flat parameters occupy a fixed range [start, start+length)
    in the corresponding buffer, so we never overwrite weights that are still
    needed by autograd for backward.
    """

    def __init__(
        self,
        buf_even: torch.Tensor | None,
        buf_odd: torch.Tensor | None,
        even_slices: dict[int, tuple[int, int]],
        odd_slices: dict[int, tuple[int, int]],
    ) -> None:
        self.buf_even = buf_even
        self.buf_odd = buf_odd
        self.even_slices = even_slices
        self.odd_slices = odd_slices

    @classmethod
    def from_block_full_sizes(
        cls,
        full_sizes: Sequence[int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "TwoBufferPool":
        """
        full_sizes[i] = total full flat numel (including padding) for block i.
        """
        even_slices: dict[int, tuple[int, int]] = {}
        odd_slices: dict[int, tuple[int, int]] = {}

        off_even = 0
        off_odd = 0
        for idx, sz in enumerate(full_sizes):
            if idx % 2 == 0:
                even_slices[idx] = (off_even, sz)
                off_even += sz
            else:
                odd_slices[idx] = (off_odd, sz)
                off_odd += sz

        buf_even = torch.empty(off_even, device=device, dtype=dtype) if off_even > 0 else None
        buf_odd = torch.empty(off_odd, device=device, dtype=dtype) if off_odd > 0 else None

        return cls(buf_even, buf_odd, even_slices, odd_slices)

    def buffer_for(self, block_idx: int) -> torch.Tensor:
        """
        Return the *block-specific* slice inside the appropriate parity buffer.
        """
        if block_idx % 2 == 0:
            if self.buf_even is None:
                raise RuntimeError("No even blocks but buffer_for called with even block_idx")
            start, length = self.even_slices[block_idx]
            return self.buf_even.narrow(0, start, length)
        else:
            if self.buf_odd is None:
                raise RuntimeError("No odd blocks but buffer_for called with odd block_idx")
            start, length = self.odd_slices[block_idx]
            return self.buf_odd.narrow(0, start, length)
