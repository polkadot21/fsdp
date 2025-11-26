import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from torch.profiler import record_function

from fsdp.buffer_pool import StaticBufferPool
from fsdp.config import Setup
from fsdp.streams import StreamManager


class GateGradFlow(torch.autograd.Function):
    """
    It acts as a synchronization fence in the Autograd graph.
    - Forward: No-op (Pass-through).
    - Backward: Guarantees that 'hook' (ReduceScatter) runs ONLY AFTER
      the gradients for the entire module have been computed.

    This prevents the race condition where a standard backward_hook might fire
    before weight gradients (p.grad) are fully populated by the C++ engine.
    """

    @staticmethod
    def forward(ctx, module, x):
        ctx.module = module
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # This runs AFTER the module's backward logic (chain rule).
        ctx.module._reduce_gradients()
        return None, grad_output


class FSDPLayer(nn.Module):
    """
    A FSDP Wrapper for a single Layer (e.g. Transformer Block).

    Key Features:
      1. **Static Memory**: Uses a fixed 'Ping-Pong' buffer pool to eliminate allocation overhead.
      2. **Stream Pipelining**: Explicitly overlaps Compute (FWD/BWD) with Comm (AG/RS).
      3. **Robust Hooks**: Handles PyTorch's internal view creation and Python's GC/ID-reuse quirks.

    The Life Cycle of a Parameter:
      - **Init**: Flattened, Sharded, and original storage is freed (p.data = empty).
      - **Prefetch**: Shard gathered from all ranks into the global buffer.
      - **Materialize**: p.data points to the buffer. We register its ID to catch it in hooks.
      - **Forward**: 'pack_hook' intercepts the save. Returns a ticket (index) instead of data.
      - **Release**: p.data freed. Buffer marked available for next layer.
      - **Backward**: 'unpack_hook' uses ticket to refetch data (Materialize).
      - **Reduce**: Gradients accumulated and scattered back to shards.
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

        self.log = logger.bind(layer=self.block_idx)

        # ---------------------------------------------------------------------
        # 1. Deterministic Parameter Order
        # ---------------------------------------------------------------------
        # CRITICAL: Pytorch's .parameters() iteration order is usually deterministic
        # but not guaranteed across different environments/versions.
        # We sort by name to ensure every GPU flattens weights in the EXACT same order.
        # If Rank 0 puts [W_q, W_k] and Rank 1 puts [W_k, W_q], the model is destroyed.
        named_sorted = sorted(self.module.named_parameters())
        self.params = [p for n, p in named_sorted]
        self.param_names = [n for n, p in named_sorted]

        # ---------------------------------------------------------------------
        # 2. Memory Safety Mechanisms
        # ---------------------------------------------------------------------
        # _tensor_id_to_index: Maps id(tensor_object) -> (block, index, is_transposed)
        # Used by _pack_hook to identify which parameter Autograd is trying to save.
        self._tensor_id_to_index = {}

        # _keep_alive: A list to hold strong references to view tensors created in _materialize.
        # WHY? If we don't keep them, Python's GC deletes the view object immediately after
        # assignment. Python then reuses that memory address (ID) for a new tensor (e.g. activation). # noqa
        # This causes an "ID Collision" where pack_hook thinks an activation is a weight.
        self._keep_alive = []

        # 3. Backward Prefetch Trigger
        # A callback function provided by the Train Loop.
        # When called, it tells the *previous* layer (N-1) to start AllGathering.
        self.backward_prefetch_trigger = None

        # ---------------------------------------------------------------------
        # 4. Flatten & Shard (The Core FSDP Logic)
        # ---------------------------------------------------------------------
        # We merge all small parameters into one giant flat vector to optimize NCCL bandwidth.
        # Layout: [Param1 | Param2 | Param3 | ... | Padding]
        flat_params = [p.detach().reshape(-1) for p in self.params]
        full_flat = torch.cat(flat_params)
        self.full_numel = full_flat.numel()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Calculate padding to ensure the total size is divisible by world_size
        if self.full_numel % world_size != 0:
            self.pad = world_size - (self.full_numel % world_size)
        else:
            self.pad = 0

        full_padded = torch.cat([full_flat, torch.zeros(self.pad, device=full_flat.device)])

        # We only keep the shard corresponding to our rank.
        # Memory Usage: Drops from 100% (Full Model) to 1/N (Shard).
        my_slice = full_padded.chunk(world_size)[rank].clone()
        self.shard = nn.Parameter(my_slice)

        # ---------------------------------------------------------------------
        # 5. Metadata for Reconstruction
        # ---------------------------------------------------------------------
        # We need these shapes to recreate the "Views" (fake parameters) from the flat buffer later.
        self.param_shapes = [p.shape for p in self.params]
        self.param_numels = [p.numel() for p in self.params]

        # ---------------------------------------------------------------------
        # 6. The Emptying (Shell Creation)
        # ---------------------------------------------------------------------
        # We detach the storage of the original parameters.
        # They now point to nothing (size 0). They are just "Shells" waiting to be filled.
        for p in self.params:
            p.data = torch.empty(0)

        self._is_materialized = False

        # ---------------------------------------------------------------------
        # 7. Pre-Backward Hook (The Overlap Trigger)
        # ---------------------------------------------------------------------
        # This hook runs *before* any gradient computation starts for this layer.
        # It is the earliest possible moment to signal the PREVIOUS layer (N-1)
        # to start fetching its weights.
        # Sequence:
        #   1. Layer N Pre-Backward Hook -> Trigger Fetch Layer N-1
        #   2. Layer N Backward Compute (Compute Stream) || Layer N-1 AllGather (Comm Stream)
        self.module.register_full_backward_pre_hook(self._pre_backward_hook)

    # =========================================================================
    #  Autograd Hooks
    # =========================================================================
    # In standard training, PyTorch saves a copy of the weights from the forward pass to compute gradients in the backward pass. # noqa
    # In FSDP, we cannot afford to keep those weights alive.
    # We want to delete them immediately after the forward pass (_release_buffer)
    # and fetch them again right before the backward pass (_materialize).
    def _pack_hook(self, tensor):
        """
        When Autograd tries to save a tensor, this hook steps in. It checks,
        "Is this one of my FSDP weights?"
        If yes, it says, "Don't save the 500MB tensor.
        Take this tiny tuple (block_idx, param_idx) instead."
        """
        # 1. Fast Path: The "Identity Check"
        # Ideally, the tensor Autograd wants to save is the EXACT same python object
        # we registered in _materialize(). We check its unique object ID.
        t_id = id(tensor)
        if t_id in self._tensor_id_to_index:
            # Bingo. We know exactly which parameter this is.
            # Return the "ticket" tuple: (Block 0, Param Index 5, Not Transposed)
            return self._tensor_id_to_index[t_id]

        # 2. The "Storage Rescue" Path (The Engineer's Nightmare)
        # If we are here, id(tensor) failed. Why?
        # Because PyTorch operations (like F.linear) sometimes create *temporary views*
        # internally. For example, they might create a `t.t()` (transpose) view.
        # That view is a new Python object (new ID), but it points to OUR buffer memory.
        # If we don't catch this, Autograd will save the full tensor view, causing an OOM
        # or a crash when we wipe the underlying buffer.

        if self._is_materialized:
            # Get the raw memory address of our big FSDP buffer
            buffer_storage = self.bufpool.get_buffer(self.block_idx).untyped_storage()

            # We compare the data_ptr of the underlying storage.
            if tensor.untyped_storage().data_ptr() == buffer_storage.data_ptr():
                # Yes, it's an alias of our buffer! We must claim it.
                ptr = tensor.data_ptr()

                # We have to find WHICH parameter it corresponds to.
                # We scan our params to see which one starts at the same memory address.
                # O(N) :(.
                for i, p in enumerate(self.params):
                    if p.data_ptr() == ptr:
                        # Found the match, but shapes might differ.
                        is_transposed = False

                        # 3. Transpose Detection
                        # Linear layers often save weight.T for backward.
                        # If shape is [A, B] but param is [B, A], it's a transpose.
                        if tensor.shape != p.shape:
                            if len(p.shape) == 2 and tensor.shape == p.shape[::-1]:
                                is_transposed = True
                            else:
                                # If shapes are wildly different (e.g. reshape),
                                # it's too risky to compress. Let PyTorch save the raw tensor.
                                # (This is rare in Transformers).
                                return tensor

                        self.log.warning(f"PackHook: Rescued {self.param_names[i]}")
                        # Return the ticket, but mark it as TRANSPOSED so we can recreate it later.
                        return (self.block_idx, i, int(is_transposed))

        # 4. Pass-through
        # If it's not in our ID map and not in our storage, it's a regular activation
        # (like the input 'x'). We let PyTorch save it normally.
        return tensor

    def _unpack_hook(self, packed):
        """
        This runs during the Backward Pass. Its job is to turn the ticket back into a valid tensor.
        """
        # 1. Check if this is our ticket
        # We use a tuple of length 3 as our signature.
        # (Standard tensors or other hooks won't match this).
        if isinstance(packed, tuple) and len(packed) == 3:
            block_idx, param_idx, is_transposed_int = packed

            # 2. THE MAGIC MOMENT: Materialization
            # We trigger the async communication to fetch weights from other GPUs.
            # If they are already fetching (prefetch), this just waits for them to arrive.
            # This populates 'self.params[i].data' with valid memory.
            self._materialize()

            # 3. Retrieve the Base Parameter
            p = self.params[param_idx]

            # 4. Sanity Check (The "Stale Pointer" Guard)
            # If _materialize failed or _release_buffer ran prematurely,
            # p.data might be empty (size 0). If we give this to Autograd, it crashes C++.
            if p.numel() == 0:
                name = self.param_names[param_idx]
                raise RuntimeError(f"FATAL: {name} is EMPTY in unpack!")

            # 5. Reconstruct the View
            # We grab the dense data tensor.
            out = p.data

            # 6. Re-apply Transpose
            # If pack_hook saw a transposed view, we must return a transposed view
            # so the math shapes match what Autograd expects.
            if is_transposed_int:
                out = out.t()

            return out

        # If it wasn't our ticket (e.g., it was an activation tensor), return it as-is.
        return packed

    # =========================================================================
    #  Logic: Communication & Synchronization
    # =========================================================================
    def prefetch_forward(self, phase="FWD"):
        """Async AllGather."""
        if not self.cfg.overlap:
            return

        with torch.cuda.stream(self.streams.comm_stream):
            free_evt = self.bufpool.get_free_event(self.block_idx)
            self.streams.comm_stream.wait_event(free_evt)

            target_buf = self.bufpool.get_buffer(self.block_idx)

            with record_function(f"AG ({phase}) Block {self.block_idx}"):
                dist.all_gather_into_tensor(
                    target_buf[: self.shard.numel() * dist.get_world_size()],
                    self.shard,
                    group=self.group,
                )

            ready_evt = self.bufpool.get_ready_event(self.block_idx)
            ready_evt.record(self.streams.comm_stream)

    def prefetch_backward(self):
        """Alias with explicit phase."""
        self.prefetch_forward(phase="BWD")

    def _materialize(self):
        if self._is_materialized:
            return

        with record_function(f"Materialize Wait {self.block_idx}"):
            if self.cfg.overlap:
                ready_evt = self.bufpool.get_ready_event(self.block_idx)
                self.streams.compute_stream.wait_event(ready_evt)
            else:
                target_buf = self.bufpool.get_buffer(self.block_idx)
                dist.all_gather_into_tensor(
                    target_buf[: self.shard.numel() * dist.get_world_size()],
                    self.shard,
                    group=self.group,
                )

        self._tensor_id_to_index.clear()
        self._keep_alive.clear()

        full_params = self.bufpool.get_buffer(self.block_idx)
        offset = 0

        for i, (p, name, numel, shape) in enumerate(  # noqa
            zip(self.params, self.param_names, self.param_numels, self.param_shapes, strict=False)
        ):
            v = full_params[offset : offset + numel].view(shape)
            p.data = v
            self._keep_alive.append(v)

            self._tensor_id_to_index[id(p)] = (self.block_idx, i, 0)
            self._tensor_id_to_index[id(v)] = (self.block_idx, i, 0)
            offset += numel

        self._is_materialized = True

    def _release_buffer(self):
        free_evt = self.bufpool.get_free_event(self.block_idx)
        free_evt.record(self.streams.compute_stream)
        for p in self.params:
            p.data = torch.empty(0)
        self._tensor_id_to_index.clear()
        self._keep_alive.clear()
        self._is_materialized = False

    # =========================================================================
    #  Execution Flow
    # =========================================================================
    def forward(self, x):
        self._materialize()

        with record_function(f"FWD Block {self.block_idx}"):
            with torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook):
                # The graph looks like: Input -> [Gate Node] -> [Module] -> Output.
                # In backward: Gradients flow from Output backwards.
                # They hit [Module] first. PyTorch executes the module's backward function (computing p.grad for weights). # noqa
                # Only after the module is finished does the gradient reach the [Gate Node].
                # This triggers GateGradFlow.backward().

                x = GateGradFlow.apply(self, x)
                out = self.module(x)

        self._release_buffer()
        return out

    def set_backward_prefetch_trigger(self, trigger_fn):
        self.backward_prefetch_trigger = trigger_fn

    def _pre_backward_hook(self, module, grad_output):
        with record_function(f"BWD Pre Block {self.block_idx}"):
            if self.cfg.overlap and self.backward_prefetch_trigger:
                self.backward_prefetch_trigger()

    def _reduce_gradients(self):
        with record_function(f"BWD Post / RS Block {self.block_idx}"):
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
            full_grad = torch.cat([g.reshape(-1) for g in grads])
            for p in self.params:
                p.grad = None

            if self.pad > 0:
                full_grad = torch.cat([full_grad, torch.zeros(self.pad, device=full_grad.device)])

            evt = torch.cuda.Event()
            evt.record(self.streams.compute_stream)
            self.streams.comm_stream.wait_event(evt)

            with torch.cuda.stream(self.streams.comm_stream):
                if self.shard.grad is None:
                    self.shard.grad = torch.zeros_like(self.shard)

                dist.reduce_scatter_tensor(
                    self.shard.grad, full_grad, op=dist.ReduceOp.AVG, group=self.group
                )

        self._release_buffer()
