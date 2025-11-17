def analyze_profiler(prof, rank):
    """
    Summary of compute vs comm vs overlap.
    """

    print(f"\n=== PROFILER SUMMARY (rank={rank}) ===")

    events = prof.key_averages(group_by_input_shape=False)

    compute_time = 0.0
    comm_time = 0.0

    compute_ops = ("mm", "addmm", "gelu", "relu", "matmul", "softmax")
    comm_ops = ("all_gather", "reduce_scatter", "all_reduce")

    # Aggregate times
    for evt in events:
        name = evt.key.lower()

        cuda_time = getattr(evt, "self_cuda_time_total", 0.0)

        if any(op in name for op in comm_ops):
            comm_time += cuda_time
        elif any(op in name for op in compute_ops):
            compute_time += cuda_time

    # Convert µs → ms
    compute_ms = compute_time / 1000.0
    comm_ms = comm_time / 1000.0
    total_ms = compute_ms + comm_ms

    print(f"Compute time (CUDA):     {compute_ms:8.2f} ms")
    print(f"Comm time (NCCL):        {comm_ms:8.2f} ms")
    print(f"Total (compute+comm):    {total_ms:8.2f} ms")

    # Compute overlap efficiency (rough estimate)
    # overlap = how much comm is hidden under compute (bounded by compute time)
    hidden_comm = min(compute_ms, comm_ms)
    visible_comm = max(0, comm_ms - compute_ms)
    overlap_ratio = 100 * (hidden_comm / comm_ms) if comm_ms > 0 else 0

    print(f"\nHidden communication:    {hidden_comm:8.2f} ms")
    print(f"Visible communication:   {visible_comm:8.2f} ms")
    print(f"Overlap ratio:           {overlap_ratio:8.1f}%")

    print("\nBreakdown of top comm kernels:")
    comm_events = [e for e in events if any(op in e.key.lower() for op in comm_ops)]

    comm_events = sorted(comm_events, key=lambda e: e.self_cuda_time_total, reverse=True)
    for e in comm_events[:5]:
        print(f"  {e.key:30s} {e.self_cuda_time_total/1000.0:8.2f} ms")

    print("=== END SUMMARY ===\n")


def print_ascii_gantt(prof, rank, width=60):
    """
    Build an ASCII Gantt chart for compute vs comm vs overlap.
    width = number of characters for the timeline.
    """

    events = prof.key_averages()

    # Define what we consider compute vs comm
    compute_ops = ("mm", "addmm", "matmul", "softmax", "gelu", "relu")
    comm_ops = ("all_gather", "reduce_scatter", "all_reduce")

    compute_us = 0.0
    comm_us = 0.0

    for e in events:
        name = e.key.lower()

        cuda_time = getattr(e, "self_cuda_time_total", 0.0)

        if any(op in name for op in compute_ops):
            compute_us += cuda_time
        elif any(op in name for op in comm_ops):
            comm_us += cuda_time

    compute_ms = compute_us / 1000
    comm_ms = comm_us / 1000
    total_ms = compute_ms + comm_ms

    def scale(x):
        return int(width * (x / total_ms)) if total_ms > 0 else 0

    # Gantt components
    comp_bar = scale(compute_ms)
    overlap_ms = min(compute_ms, comm_ms)
    exposed_ms = max(0.0, comm_ms - compute_ms)

    overlap_bar = scale(overlap_ms)
    exposed_bar = scale(exposed_ms)

    print(f"\n=== ASCII GANTT TIMELINE (rank={rank}) ===")
    print(f"Total step time ≈ {total_ms:.2f} ms\n")

    print(" Legend:")
    print("   █ = compute")
    print("   ▒ = overlapped comm (hidden by compute)")
    print("   ░ = exposed comm (stall)\n")

    # Build rows
    compute_row = "█" * comp_bar
    comm_row = "▒" * overlap_bar + "░" * exposed_bar

    print(" Compute:   " + compute_row)
    print(" NCCL:      " + comm_row)

    if comm_ms > 0:
        overlap_ratio = overlap_ms / comm_ms
    else:
        overlap_ratio = 1.0

    print(f"\n Overlap ratio: {overlap_ratio*100:.1f}%")
    print("===========================================\n")
