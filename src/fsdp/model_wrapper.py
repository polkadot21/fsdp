import torch
import torch.nn as nn

from fsdp.buffer_pool import StaticBufferPool
from fsdp.config import Setup
from fsdp.fsdp_layer import FSDPLayer
from fsdp.models.tiny_model import TinyModel
from fsdp.streams import StreamManager


class FSDPWrapper(nn.Module):
    def __init__(self, cfg: Setup, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # 1. Create Base Model
        base = TinyModel(cfg.in_dim, cfg.dim, cfg.n_heads, cfg.ff_dim, cfg.n_layers).to(device)

        # 2. Setup Shared Components
        self.streams = StreamManager(device)

        # Determine Buffer Size (Max parameter count of any single block)
        sizes = [sum(p.numel() for p in blk.parameters()) for blk in base.blocks]
        self.bufpool = StaticBufferPool.from_block_sizes(sizes, device, torch.float32)

        # 3. Wrap Blocks
        self.layers = nn.ModuleList()
        for i, blk in enumerate(base.blocks):
            self.layers.append(FSDPLayer(blk, i, self.streams, self.bufpool, cfg))

        self.inp = base.inp
        self.out = base.out

    def forward(self, x):
        x = self.inp(x)

        # --- Forward Pass ---
        # Prime the pump: Prefetch Block 0
        self.layers[0].prefetch_forward()

        for i, layer in enumerate(self.layers):
            # Pipeline: While we compute i, prefetch i+1
            if i + 1 < len(self.layers):
                self.layers[i + 1].prefetch_forward()

            x = layer(x)

        out = self.out(x)

        # --- Backward Pass Prefetching Logic ---
        # Standard Autograd doesn't let us easily insert "Prefetch i-1" inside "Backward i".
        # However, we can use the `register_full_backward_hook` on Layer i to trigger prefetch for i-1. # noqa
        # We inject a callback into the layers now that we have the list.
        return out
