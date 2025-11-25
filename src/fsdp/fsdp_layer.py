import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from fsdp.buffer_pool import StaticBufferPool
from fsdp.config import Setup
from fsdp.streams import StreamManager


class FSDPLayer(nn.Module):
    """
    A Fully Sharded Data Parallel (FSDP) wrapper for a single Transformer Block.

    Architecture:
      - Memory: Uses a static 'Ping-Pong' buffer pool. Two fixed-size buffers are reused
        across all layers to hold materialized (full) weights.
      - Streams: Explicitly manages 'Compute' vs 'Communication' CUDA streams.
      - Consistency: Uses Autograd hooks to prevent 'stale pointer' bugs where gradients
        would be computed on overwritten buffers.

    Flow:
      1. Prefetch (Comm Stream): Async AllGather into Buffer X.
      2. Materialize (Compute Stream): Wait for Comm, point params to Buffer X.
      3. Forward (Compute Stream): Run module, intercept Autograd saves.
      4. Release (Compute Stream): Mark Buffer X as free for Comm to overwrite later.
    """

    def __init__(
        self,
        module: nn.Module,
        block_idx: int,
        streams: StreamManager,
        bufpool: StaticBufferPool,
        cfg: Setup,
        group: dist.ProcessGroup = None,
    ):
        super().__init__()
        self.module = module
        self.block_idx = block_idx
        self.streams = streams
        self.bufpool = bufpool
        self.cfg = cfg
        self.group = group

        # --- LOGGING SETUP ---
        # Create a logger bound to this specific layer index.
        # This ensures logs look like: "2024-11-25 10:00:00 | DEBUG | Layer 3 | prefetch_forward: ..." # noqa
        self.log = logger.bind(layer=self.block_idx)

        # ---------------------------------------------------------------------
        # 1. Deterministic Parameter Order
        # ---------------------------------------------------------------------
        # CRITICAL: We strictly sort parameters by name.
        self.params = [p for n, p in sorted(self.module.named_parameters())]

        # ---------------------------------------------------------------------
        # 2. Map for Hooks
        # ---------------------------------------------------------------------
        for i, p in enumerate(self.params):
            p._fsdp_tag = (self.block_idx, i)

        # ---------------------------------------------------------------------
        # 3. Flatten & Shard
        # ---------------------------------------------------------------------
        flat_params = [p.detach().reshape(-1) for p in self.params]
        full_flat = torch.cat(flat_params)
        self.full_numel = full_flat.numel()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if self.full_numel % world_size != 0:
            self.pad = world_size - (self.full_numel % world_size)
        else:
            self.pad = 0

        # Create padded full vector
        full_padded = torch.cat([full_flat, torch.zeros(self.pad, device=full_flat.device)])

        # Extract MY slice (the only thing we keep in persistent VRAM)
        my_slice = full_padded.chunk(world_size)[rank].clone()
        self.shard = nn.Parameter(my_slice)

        # ---------------------------------------------------------------------
        # 4. Metadata
        # ---------------------------------------------------------------------
        self.param_shapes = [p.shape for p in self.params]
        self.param_numels = [p.numel() for p in self.params]

        # ---------------------------------------------------------------------
        # 5. Free Original Parameters (The "Emptying")
        # ---------------------------------------------------------------------
        for p in self.params:
            p.data = torch.empty(0)

        self._is_materialized = False

        # Register the hook that runs AFTER the backward pass of this module finishes.
        self.module.register_full_backward_hook(self._post_backward_hook)

    # =========================================================================
    #  Autograd Hooks
    # =========================================================================
    def _pack_hook(self, tensor):
        self.log.debug("_pack_hook called for tensor")
        if hasattr(tensor, "_fsdp_tag"):
            self.log.debug(f"_pack_hook found tag: {tensor._fsdp_tag}")
            return tensor._fsdp_tag
        return tensor

    def _unpack_hook(self, packed):
        self.log.debug("_unpack_hook called for params")

        if isinstance(packed, tuple) and len(packed) == 2:
            # 1. Resurrect
            self._materialize()

            # 2. Retrieve
            block_idx, param_idx = packed
            self.log.debug(f"_unpack_hook called for block: {block_idx}, param {param_idx}")

            p = self.params[param_idx]

            # 3. Verify
            if p.numel() == 0:
                self.log.critical(f"FATAL: Param {param_idx} is EMPTY in unpack_hook!")
                raise RuntimeError("Parameter failed to materialize!")

            return p
        return packed

    # =========================================================================
    #  Logic: Communication & Synchronization
    # =========================================================================

    def prefetch_forward(self):
        """
        Launches the Async AllGather on the Communication Stream.
        Designed to be called BEFORE this layer is needed (Overlap).
        """
        if not self.cfg.overlap:
            return

        self.log.debug("prefetch_forward: Waiting for buffer release (compute -> comm)")

        # [Stream: Comm]
        with torch.cuda.stream(self.streams.comm_stream):
            # 1. Wait for Compute to finish using this buffer from the PREVIOUS iteration.
            free_evt = self.bufpool.get_free_event(self.block_idx)
            self.streams.comm_stream.wait_event(free_evt)

            self.log.debug("prefetch_forward: Launching AllGather")

            # 2. NCCL AllGather
            target_buf = self.bufpool.get_buffer(self.block_idx)
            dist.all_gather_into_tensor(
                target_buf[: self.shard.numel() * dist.get_world_size()],
                self.shard,
                group=self.group,
            )

            # 3. Record "Data Ready" Event
            ready_evt = self.bufpool.get_ready_event(self.block_idx)
            ready_evt.record(self.streams.comm_stream)

            self.log.debug("prefetch_forward: Done. Event Recorded.")

    def prefetch_backward(self):
        """Alias for clarity. The operation (AllGather) is identical."""
        self.log.debug("prefetch_backward: Triggered")
        self.prefetch_forward()

    def _materialize(self):
        """
        Ensures parameter pointers (p.data) are valid and pointing to the buffer.
        """
        if self._is_materialized:
            return

        self.log.debug("Materialize: Waiting for Data Ready event (comm -> compute)...")

        # 1. Synchronization
        if self.cfg.overlap:
            # [Stream: Compute]
            ready_evt = self.bufpool.get_ready_event(self.block_idx)
            self.streams.compute_stream.wait_event(ready_evt)
        else:
            # Sync Fallback
            target_buf = self.bufpool.get_buffer(self.block_idx)
            dist.all_gather_into_tensor(
                target_buf[: self.shard.numel() * dist.get_world_size()],
                self.shard,
                group=self.group,
            )

        # 2. Pointer Arithmetic (Resurrection)
        full_params = self.bufpool.get_buffer(self.block_idx)
        offset = 0
        for p, numel, shape in zip(self.params, self.param_numels, self.param_shapes, strict=False):
            p.data = full_params[offset : offset + numel].view(shape)
            offset += numel

        self._is_materialized = True
        self.log.debug("Materialize: Complete. Params are now valid views.")

    def _release_buffer(self):
        """
        Mark the buffer as 'Free' so the Comm stream can use it for the next layer.
        """
        self.log.debug("Release Buffer: Marking as free.")

        # [Stream: Compute]
        free_evt = self.bufpool.get_free_event(self.block_idx)
        free_evt.record(self.streams.compute_stream)

        # Safety: Set pointers to None/Empty so no one accesses stale data by accident.
        for p in self.params:
            p.data = torch.empty(0)

        self._is_materialized = False

    # =========================================================================
    #  Execution Flow
    # =========================================================================

    def forward(self, x):
        """
        Standard Forward Pass with Hook Injection.
        """
        # 1. Ensure weights are present (Wait for Prefetch)
        self._materialize()

        # Optional: Trace log for heavy debugging
        # self.log.trace("Forward: Entering Module")

        # 2. Run Forward inside the Hook Context
        with torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook):
            out = self.module(x)

        # 3. Release Buffer immediately
        self._release_buffer()
        return out

    def _post_backward_hook(self, module, gin, gout):
        """
        Triggered automatically by Autograd after gradients for this block are computed.
        """
        self.log.debug("Post-Backward: Hook triggered. Flattening grads.")

        # ---------------------------------------------------------------------
        # Step A: Flatten Gradients
        # ---------------------------------------------------------------------
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        full_grad = torch.cat([g.reshape(-1) for g in grads])

        # Free the PyTorch grad tensors immediately to save memory
        for p in self.params:
            p.grad = None

        # Handle Padding
        if self.pad > 0:
            full_grad = torch.cat([full_grad, torch.zeros(self.pad, device=full_grad.device)])

        # ---------------------------------------------------------------------
        # Step B: Synchronization (Compute -> Comm)
        # ---------------------------------------------------------------------
        self.log.debug("Post-Backward: Syncing (compute -> comm) for ReduceScatter")
        evt = torch.cuda.Event()
        evt.record(self.streams.compute_stream)
        self.streams.comm_stream.wait_event(evt)

        # ---------------------------------------------------------------------
        # Step C: ReduceScatter (Comm Stream)
        # ---------------------------------------------------------------------
        with torch.cuda.stream(self.streams.comm_stream):
            if self.shard.grad is None:
                self.shard.grad = torch.zeros_like(self.shard)

            dist.reduce_scatter_tensor(
                self.shard.grad, full_grad, op=dist.ReduceOp.SUM, group=self.group
            )

        # ---------------------------------------------------------------------
        # Step D: Cleanup
        # ---------------------------------------------------------------------
        self._release_buffer()
        self.log.debug("Post-Backward: Complete. Buffer released.")

        return None
