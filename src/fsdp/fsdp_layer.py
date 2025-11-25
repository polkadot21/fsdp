import torch
import torch.distributed as dist
import torch.nn as nn

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

        # ---------------------------------------------------------------------
        # 1. Deterministic Parameter Order
        # ---------------------------------------------------------------------
        # CRITICAL: We strictly sort parameters by name.
        # Why? If Rank 0 flattens [w, b] and Rank 1 flattens [b, w],
        # the AllGather reconstruction will be garbage.
        self.params = [p for n, p in sorted(self.module.named_parameters())]

        # ---------------------------------------------------------------------
        # 2. Map for Hooks
        # ---------------------------------------------------------------------
        # We need a fast O(1) way to identify if a tensor Autograd is saving
        # is one of OUR managed FSDP parameters or just a random activation.
        # id(tensor) -> index in self.params list
        self.param_id_to_index = {id(p): i for i, p in enumerate(self.params)}

        # ---------------------------------------------------------------------
        # 3. Flatten & Shard
        # ---------------------------------------------------------------------
        # We merge all small parameters (weights, biases) into one giant vector.
        # Data Flow:
        #   [p1: (10,10)] [p2: (10,)]  ->  [p_flat: (110,)]
        flat_params = [p.detach().reshape(-1) for p in self.params]
        full_flat = torch.cat(flat_params)
        self.full_numel = full_flat.numel()

        # Sharding Logic:
        #   Total Params: 100
        #   World Size: 4
        #   Shard Size: 25 (Everyone keeps 1/4th)
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
        # We need these to reconstruct the "Views" (fake tensors) from the flat buffer later.
        self.param_shapes = [p.shape for p in self.params]
        self.param_numels = [p.numel() for p in self.params]

        # ---------------------------------------------------------------------
        # 5. Free Original Parameters (The "Emptying")
        # ---------------------------------------------------------------------
        # We detach the storage of the original parameters.
        # They now point to nothing (size 0). They are just empty shells.
        # We will dynamically fill them with data pointers during forward/backward.
        for p in self.params:
            p.data = torch.empty(0)

        self._is_materialized = False

        # Register the hook that runs AFTER the backward pass of this module finishes.
        self.module.register_full_backward_hook(self._post_backward_hook)

    # =========================================================================
    #  Autograd Hooks
    # =========================================================================
    """
    DATA FLOW:
    Forward Pass:
      Autograd: "I need to save 'Weight Matrix A' for the backward pass."
      _pack_hook: "Stop! 'Weight Matrix A' is in a volatile buffer. It will be overwritten."
                  "Here, take this ticket instead: (Block 5, Param Index 3)."
      Autograd: Saves (5, 3).

    Backward Pass:
      Autograd: "I need the tensor for ticket (5, 3)."
      _unpack_hook: "Hold on." -> Calls _materialize() -> "Buffer refilled."
                    -> "Here is 'Weight Matrix A' (view into fresh buffer)."
    """

    def _pack_hook(self, tensor):
        idx = self.param_id_to_index.get(id(tensor), None)
        if idx is not None:
            return (self.block_idx, idx)
        return tensor

    def _unpack_hook(self, packed):
        if isinstance(packed, tuple) and len(packed) == 2:
            self._materialize()
            return self.params[packed[1]]
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

        # [Stream: Comm]
        with torch.cuda.stream(self.streams.comm_stream):
            # 1. Wait for Compute to finish using this buffer from the PREVIOUS iteration.
            #    We cannot overwrite the buffer if the backward pass of 2 steps ago is still reading it. # noqa
            #    (Though typically for forward, we wait on the 'free' event recorded after forward release) # noqa
            free_evt = self.bufpool.get_free_event(self.block_idx)
            self.streams.comm_stream.wait_event(free_evt)

            # 2. NCCL AllGather
            #    Gather all shards -> Reconstruct full weights in Global Buffer
            target_buf = self.bufpool.get_buffer(self.block_idx)
            dist.all_gather_into_tensor(
                target_buf[: self.shard.numel() * dist.get_world_size()],
                self.shard,
                group=self.group,
            )

            # 3. Record "Data Ready" Event
            #    "Hey Compute Stream, the weights are now safe to read!"
            ready_evt = self.bufpool.get_ready_event(self.block_idx)
            ready_evt.record(self.streams.comm_stream)

    def prefetch_backward(self):
        """Alias for clarity. The operation (AllGather) is identical."""
        self.prefetch_forward()

    def _materialize(self):
        """
        Ensures parameter pointers (p.data) are valid and pointing to the buffer.
        If async overlap is on, this acts as a synchronization barrier for the Compute Stream.
        """
        if self._is_materialized:
            return

        # 1. Synchronization
        if self.cfg.overlap:
            # [Stream: Compute]
            # Wait until the "Data Ready" event recorded by the Comm stream fires.
            # CPU does NOT wait here. Only the GPU Compute Queue stalls.
            ready_evt = self.bufpool.get_ready_event(self.block_idx)
            self.streams.compute_stream.wait_event(ready_evt)
        else:
            # Sync Fallback: Do it right now, blocking everything.
            target_buf = self.bufpool.get_buffer(self.block_idx)
            dist.all_gather_into_tensor(
                target_buf[: self.shard.numel() * dist.get_world_size()],
                self.shard,
                group=self.group,
            )

        # 2. Pointer Arithmetic (Resurrection)
        #    We take the flat buffer and slice it up into views.
        #    We assign these views to .data of our empty parameter shells.
        full_params = self.bufpool.get_buffer(self.block_idx)
        offset = 0
        for p, numel, shape in zip(self.params, self.param_numels, self.param_shapes, strict=False):
            # SAFETY: .view() creates a reference, not a copy. Zero memory overhead.
            p.data = full_params[offset : offset + numel].view(shape)
            offset += numel

        self._is_materialized = True

    def _release_buffer(self):
        """
        Mark the buffer as 'Free' so the Comm stream can use it for the next layer.
        """
        # [Stream: Compute]
        # Record "Finished Using" event.
        # "Hey Comm Stream, I'm done with these weights. You can overwrite them now."
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

        # 2. Run Forward inside the Hook Context
        #    This ensures Autograd saves our 'tickets' (indices) instead of
        #    pointers to the buffer we are about to release.
        with torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook):
            out = self.module(x)

        # 3. Release Buffer immediately
        #    We don't need the weights anymore. Backward pass will fetch them again.
        self._release_buffer()
        return out

    def _post_backward_hook(self, module, gin, gout):
        """
        Triggered automatically by Autograd after gradients for this block are computed.
        We must Flatten Grads -> ReduceScatter -> Free Grads.
        """
        # ---------------------------------------------------------------------
        # Step A: Flatten Gradients
        # ---------------------------------------------------------------------
        # At this point, p.grad contains the full gradients (accumulated on the buffer).
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
        # Comm stream must wait for Compute stream to FINISH creating 'full_grad'.
        # We cannot ReduceScatter a tensor that is still being written to.
        evt = torch.cuda.Event()
        evt.record(self.streams.compute_stream)
        self.streams.comm_stream.wait_event(evt)

        # ---------------------------------------------------------------------
        # Step C: ReduceScatter (Comm Stream)
        # ---------------------------------------------------------------------
        # Collapse full gradients across all ranks into the local shard.
        with torch.cuda.stream(self.streams.comm_stream):
            if self.shard.grad is None:
                self.shard.grad = torch.zeros_like(self.shard)

            # Op: Sum gradients from all GPUs, then split result among GPUs.
            dist.reduce_scatter_tensor(
                self.shard.grad, full_grad, op=dist.ReduceOp.SUM, group=self.group
            )

        # ---------------------------------------------------------------------
        # Step D: Cleanup
        # ---------------------------------------------------------------------
        # We materialized the weights for backward (via unpack_hook).
        # Now that backward is done, release the buffer again.
        self._release_buffer()

        return None
