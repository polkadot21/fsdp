import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from fsdp.buffer_pool import StaticBufferPool
from fsdp.config import Setup
from fsdp.streams import StreamManager


class FSDPLayer(nn.Module):
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

        # 1. Deterministic Parameter Order
        named_sorted = sorted(self.module.named_parameters())
        self.params = [p for n, p in named_sorted]
        self.param_names = [n for n, p in named_sorted]

        # 2. Dynamic ID Map & Keep Alive
        # _tensor_id_to_index: Maps id(tensor) -> (block, idx, is_transposed)
        # _keep_alive: Holds references to view tensors to prevent Python ID reuse
        self._tensor_id_to_index = {}
        self._keep_alive = []

        # 3. Flatten & Shard
        flat_params = [p.detach().reshape(-1) for p in self.params]
        full_flat = torch.cat(flat_params)
        self.full_numel = full_flat.numel()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if self.full_numel % world_size != 0:
            self.pad = world_size - (self.full_numel % world_size)
        else:
            self.pad = 0

        full_padded = torch.cat([full_flat, torch.zeros(self.pad, device=full_flat.device)])
        my_slice = full_padded.chunk(world_size)[rank].clone()
        self.shard = nn.Parameter(my_slice)

        # 4. Metadata
        self.param_shapes = [p.shape for p in self.params]
        self.param_numels = [p.numel() for p in self.params]

        # 5. Empty Original Parameters
        for p in self.params:
            p.data = torch.empty(0)

        self._is_materialized = False
        self.module.register_full_backward_hook(self._post_backward_hook)

    # =========================================================================
    #  Autograd Hooks
    # =========================================================================
    def _pack_hook(self, tensor):
        # 1. Fast Path: ID Lookup
        t_id = id(tensor)
        if t_id in self._tensor_id_to_index:
            return self._tensor_id_to_index[t_id]

        # 2. Storage Rescue (Fallback)
        if self._is_materialized:
            buffer_storage = self.bufpool.get_buffer(self.block_idx).untyped_storage()
            if tensor.untyped_storage().data_ptr() == buffer_storage.data_ptr():
                ptr = tensor.data_ptr()
                for i, p in enumerate(self.params):
                    if p.data_ptr() == ptr:
                        is_transposed = False
                        if tensor.shape != p.shape:
                            if len(p.shape) == 2 and tensor.shape == p.shape[::-1]:
                                is_transposed = True
                            else:
                                return tensor  # Shape mismatch beyond simple transpose

                        # self.log.warning(f"PackHook: Rescued {self.param_names[i]}")
                        return (self.block_idx, i, int(is_transposed))

        return tensor

    def _unpack_hook(self, packed):
        if isinstance(packed, tuple) and len(packed) == 3:
            block_idx, param_idx, is_transposed_int = packed

            # 1. Resurrect
            self._materialize()

            # 2. Retrieve
            p = self.params[param_idx]

            # 3. Sanity Check
            if p.numel() == 0:
                name = self.param_names[param_idx]
                raise RuntimeError(f"FATAL: {name} (Param {param_idx}) is EMPTY in unpack!")

            # 4. Handle Transpose
            out = p.data
            if is_transposed_int:
                out = out.t()
            return out

        return packed

    # =========================================================================
    #  Logic
    # =========================================================================
    def prefetch_forward(self):
        if not self.cfg.overlap:
            return
        with torch.cuda.stream(self.streams.comm_stream):
            free_evt = self.bufpool.get_free_event(self.block_idx)
            self.streams.comm_stream.wait_event(free_evt)

            target_buf = self.bufpool.get_buffer(self.block_idx)
            dist.all_gather_into_tensor(
                target_buf[: self.shard.numel() * dist.get_world_size()],
                self.shard,
                group=self.group,
            )
            ready_evt = self.bufpool.get_ready_event(self.block_idx)
            ready_evt.record(self.streams.comm_stream)

    def prefetch_backward(self):
        self.prefetch_forward()

    def _materialize(self):
        if self._is_materialized:
            return

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

        # Clear old state
        self._tensor_id_to_index.clear()
        self._keep_alive.clear()

        full_params = self.bufpool.get_buffer(self.block_idx)
        offset = 0

        for i, (p, numel, shape) in enumerate(
            zip(self.params, self.param_numels, self.param_shapes, strict=False)
        ):
            # Create view
            v = full_params[offset : offset + numel].view(shape)
            p.data = v

            self._keep_alive.append(v)

            # Register IDs
            self._tensor_id_to_index[id(p)] = (self.block_idx, i, 0)
            self._tensor_id_to_index[id(v)] = (self.block_idx, i, 0)

            offset += numel

        self._is_materialized = True

    def _release_buffer(self):
        free_evt = self.bufpool.get_free_event(self.block_idx)
        free_evt.record(self.streams.compute_stream)

        for p in self.params:
            p.data = torch.empty(0)

        # Clean up to free memory and allow ID reuse SAFELY (after we are done)
        self._tensor_id_to_index.clear()
        self._keep_alive.clear()
        self._is_materialized = False

    def forward(self, x):
        self._materialize()
        with torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook):
            out = self.module(x)
        self._release_buffer()
        return out

    def _post_backward_hook(self, module, gin, gout):
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
                self.shard.grad, full_grad, op=dist.ReduceOp.SUM, group=self.group
            )

        self._release_buffer()
        return None
