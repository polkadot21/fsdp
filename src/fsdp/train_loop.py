import torch
import torch.distributed as dist
import torch.optim as optim
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function

from fsdp.model_wrapper import FSDPWrapper


def link_backward_prefetching(model: FSDPWrapper):
    layers = model.layers
    for i in range(len(layers)):
        # If I am Layer i, I want to trigger prefetch for Layer i-1
        if i > 0:
            # Define the trigger
            def trigger(prev_idx=i - 1):
                layers[prev_idx].prefetch_backward()

            # Register it on Layer i
            layers[i].set_backward_prefetch_trigger(trigger)


def train_worker(rank, world_size, cfg):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    if rank == 0:
        logger.info(
            f"Starting Worker | Model: {cfg.train.model_type} | Overlap: {cfg.train.overlap}"
        )

    model = FSDPWrapper(cfg.train, device)

    # Enable Backward Pipelining
    if cfg.train.overlap:
        link_backward_prefetching(model)

    # Collect shards for optimizer
    params = list(model.inp.parameters()) + list(model.out.parameters())
    for layer in model.layers:
        params.append(layer.shard)

    logger.debug(f"Rank {rank} | Optimizer params collected: {len(params)} tensors")

    optimizer = optim.AdamW(params, lr=cfg.train.lr)

    # Dummy Data
    B, T, D = cfg.train.batch, cfg.train.T, cfg.train.in_dim

    # -------------------------------------------------------------------------
    # 2. Warmup
    # -------------------------------------------------------------------------
    logger.info(f"Rank {rank} | Starting Warmup (3 steps)...")
    for _ in range(3):
        x = torch.randn(B, T, D, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    dist.barrier()
    logger.info(f"Rank {rank} | Warmup Complete.")

    # Sanity Check Shapes
    logger.info("Verifying Shapes...")
    model.layers[0]._materialize()
    for n, p in model.layers[0].named_parameters():
        logger.info(f"{n}: {p.shape}")

    # -------------------------------------------------------------------------
    # 3. Profiling Loop
    # -------------------------------------------------------------------------
    # Only Rank 0 prints the table header
    if rank == 0:
        logger.info(f"{'Step':<6} | {'Loss':<10} | {'VRAM (MB)':<10} | {'Phase':<10}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for s in range(cfg.profiler.n_steps):
            # Memory Snapshot (Pre-step)
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

            with record_function(f"Step {s}"):
                # --- A. FORWARD ---
                # logger.debug(f"Step {s} | Phase: Forward Start")
                x = torch.randn(B, T, D, device=device)
                y = model(x)
                loss = y.sum()

                # Check for Stale Pointer Bug (Loss Explosion/Stagnation)
                loss_val = loss.item()
                if rank == 0:
                    logger.info(f"{s:<6} | {loss_val:<10.4f} | {mem_mb:<10.1f} | {'FWD':<10}")

                # --- B. BACKWARD ---
                # logger.debug(f"Step {s} | Phase: Backward Start")

                # Prime the pump for Backward: Prefetch last layer manually
                if cfg.train.overlap:
                    # logger.trace("Triggering prefetch for LAST layer (Backward Prime)")
                    model.layers[-1].prefetch_backward()

                loss.backward()

                # --- C. OPTIMIZER ---
                logger.debug(f"Step {s} | Phase: Optimizer Step")
                if cfg.train.overlap:
                    model.wait_for_post_backward()

                optimizer.step()
                optimizer.zero_grad()

                # Sync for clean profiling steps (Optional, makes trace easier to read)
                dist.barrier()

    # -------------------------------------------------------------------------
    # 4. Teardown
    # -------------------------------------------------------------------------
    if rank == 0:
        trace_name = f"{cfg.logs_dir}/trace_{cfg.train.model_type}.json"
        prof.export_chrome_trace(trace_name)
        logger.success(f"Profiling Complete. Trace saved: {trace_name}")

    dist.destroy_process_group()
