import torch
import torch.distributed as dist
import torch.optim as optim
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function

from fsdp.model_wrapper import FSDPWrapper


def link_backward_prefetching(model: FSDPWrapper):
    """
    Links layer[i] backward hook to layer[i-1].prefetch_backward().
    This creates the daisy-chain for overlap in the backward pass.
    """
    layers = model.layers

    def get_hook(prev_layer_idx):
        def hook(module, gin, gout):
            if prev_layer_idx >= 0:
                layers[prev_layer_idx].prefetch_backward()

        return hook

    # Layer i finishing backward -> Trigger prefetch for i-1
    for i in range(len(layers)):
        # We register on the module itself
        layers[i].register_full_backward_hook(get_hook(i - 1))


def train_worker(rank, world_size, cfg):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    logger.info(f"Rank {rank} | Model: {cfg.train.model_type} | Overlap: {cfg.train.overlap}")

    model = FSDPWrapper(cfg.train, device)

    # Enable Backward Pipelining
    if cfg.train.overlap:
        link_backward_prefetching(model)

    # Collect shards for optimizer
    # (Only the shards are parameters, the original params are empty)
    params = list(model.inp.parameters()) + list(model.out.parameters())
    for layer in model.layers:
        params.append(layer.shard)

    optimizer = optim.AdamW(params, lr=cfg.train.lr)

    # Dummy Data
    B, T, D = cfg.train.batch, cfg.train.T, cfg.train.in_dim

    # Warmup
    for _ in range(3):
        x = torch.randn(B, T, D, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    dist.barrier()
    logger.info(f"Rank {rank} Warmup done. Starting Profile.")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for s in range(cfg.profiler.n_steps):
            with record_function(f"Step {s}"):
                # 1. Forward (Triggers internal prefetch chain)
                x = torch.randn(B, T, D, device=device)
                y = model(x)
                loss = y.sum()

                # 2. Backward
                # Prime the pump for Backward: Prefetch last layer manually
                if cfg.train.overlap:
                    model.layers[-1].prefetch_backward()

                loss.backward()

                # 3. Step
                optimizer.step()
                optimizer.zero_grad()

                # Sync for clean profiling steps
                dist.barrier()

    if rank == 0:
        trace_name = f"{cfg.logs_dir}/trace_{cfg.train.model_type}.json"
        prof.export_chrome_trace(trace_name)
        logger.info(f"Trace saved: {trace_name}")

    dist.destroy_process_group()
