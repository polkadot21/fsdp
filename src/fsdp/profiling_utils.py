def analyze_profiler(prof, rank: int) -> None:
    """
    Summarize compute vs comm vs overlap from a torch.profiler.Profile.
    """

    print(f"\n=== PROFILER SUMMARY (rank={rank}) ===")

    events = prof.key_averages(group_by_input_shape=False)

    compute_time_us = 0.0
    comm_time_us = 0.0
    any_cuda = False

    compute_ops = ("mm", "addmm", "gelu", "relu", "matmul", "softmax")
    comm_ops = ("all_gather", "reduce_scatter", "all_reduce")

    for evt in events:
        name = evt.key.lower()

        if hasattr(evt, "cuda_time_total"):
            cuda_time = evt.cuda_time_total
        else:
            cuda_time = getattr(evt, "self_cuda_time_total", 0.0)

        if cuda_time > 0:
            any_cuda = True
        if any(op in name for op in comm_ops):
            comm_time_us += cuda_time
        elif any(op in name for op in compute_ops):
            compute_time_us += cuda_time
    if not any_cuda:
        print(
            "No CUDA activity recorded. Are you sure the model ran on GPU "
            "and ProfilerActivity.CUDA is enabled?"
        )
        print("=== END SUMMARY ===\n")
        return

    # Convert µs → ms
    compute_ms = compute_time_us / 1000.0
    comm_ms = comm_time_us / 1000.0
    total_ms = compute_ms + comm_ms

    print(f"Compute time (CUDA):     {compute_ms:8.2f} ms")
    print(f"Comm time (NCCL):        {comm_ms:8.2f} ms")
    print(f"Total (compute+comm):    {total_ms:8.2f} ms")
    hidden_comm = min(compute_ms, comm_ms)
    visible_comm = max(0.0, comm_ms - compute_ms)
    overlap_ratio = 100.0 * (hidden_comm / comm_ms) if comm_ms > 0 else 0.0

    print(f"\nHidden communication:    {hidden_comm:8.2f} ms")
    print(f"Visible communication:   {visible_comm:8.2f} ms")
    print(f"Overlap ratio:           {overlap_ratio:8.1f}%")

    print("\nBreakdown of top comm kernels:")
    comm_events = [e for e in events if any(op in e.key.lower() for op in comm_ops)]
    comm_events = sorted(
        comm_events,
        key=lambda e: getattr(e, "cuda_time_total", getattr(e, "self_cuda_time_total", 0.0)),
        reverse=True,
    )
    for e in comm_events[:5]:
        ct = getattr(e, "cuda_time_total", getattr(e, "self_cuda_time_total", 0.0)) / 1000.0
        print(f"  {e.key:30s} {ct:8.2f} ms")

    print("=== END SUMMARY ===\n")


def print_ascii_gantt(prof, rank: int, width: int = 60) -> None:
    """
    Build an ASCII Gantt chart for compute vs comm vs overlap.
    width = number of characters for the timeline.
    """

    events = prof.key_averages(group_by_input_shape=False)

    compute_ops = ("mm", "addmm", "matmul", "softmax", "gelu", "relu")
    comm_ops = ("all_gather", "reduce_scatter", "all_reduce")

    compute_us = 0.0
    comm_us = 0.0
    any_cuda = False

    for e in events:
        name = e.key.lower()

        if hasattr(e, "cuda_time_total"):
            cuda_time = e.cuda_time_total
        else:
            cuda_time = getattr(e, "self_cuda_time_total", 0.0)

        if cuda_time > 0:
            any_cuda = True

        if any(op in name for op in compute_ops):
            compute_us += cuda_time
        elif any(op in name for op in comm_ops):
            comm_us += cuda_time

    if not any_cuda:
        print(f"\n=== ASCII GANTT TIMELINE (rank={rank}) ===")
        print("No CUDA activity recorded; skipping timeline.")
        print("===========================================\n")
        return

    compute_ms = compute_us / 1000.0
    comm_ms = comm_us / 1000.0
    total_ms = compute_ms + comm_ms

    def scale(x: float) -> int:
        return int(width * (x / total_ms)) if total_ms > 0 else 0

    comp_bar = scale(compute_ms)
    overlap_ms = min(compute_ms, comm_ms)
    exposed_ms = max(0.0, comm_ms - compute_ms)
    overlap_bar = scale(overlap_ms)
    exposed_bar = scale(exposed_ms)

    compute_row = "█" * comp_bar
    comm_row = "▒" * overlap_bar + "░" * exposed_bar

    print(f"\n=== ASCII GANTT TIMELINE (rank={rank}) ===")
    print(f"Total step time ≈ {total_ms:.2f} ms\n")

    print(" Legend:")
    print("   █ = compute")
    print("   ▒ = overlapped comm (hidden by compute)")
    print("   ░ = exposed comm (stall)\n")

    print(" Compute:   " + compute_row)
    print(" NCCL:      " + comm_row)

    overlap_ratio = (overlap_ms / comm_ms * 100.0) if comm_ms > 0 else 100.0
    print(f"\n Overlap ratio: {overlap_ratio:.1f}%")
    print("===========================================\n")
