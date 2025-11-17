# src/fsdp/profiling_utils.py

from __future__ import annotations

# --------------------------- helpers -----------------------------------------


def _get_device_times_us(evt) -> tuple[float, float]:
    """
    Return (self_device_us, inclusive_device_us) for a profiler event.

    We try the *most generic, modern* attributes first (device_time_total /
    self_device_time_total), then CUDA-specific ones, then fall back to
    per-call fields (device_time / cuda_time).

    All values are returned in microseconds (µs). If a field is missing or
    zero, we move on to the next candidate.
    """

    # ---- self device time (exclusive) ----
    self_us = 0.0

    # 1) New, device-agnostic names (preferred)
    for attr in ("self_device_time_total",):
        if hasattr(evt, attr):
            v = getattr(evt, attr)
            if v:
                self_us = float(v)
                break

    # 2) CUDA-specific self totals (older builds)
    if self_us == 0.0:
        for attr in ("self_cuda_time_total",):
            if hasattr(evt, attr):
                v = getattr(evt, attr)
                if v:
                    self_us = float(v)
                    break

    # 3) Per-call self device time (deprecated-style fields)
    if self_us == 0.0:
        for attr in ("self_device_time", "self_cuda_time"):
            if hasattr(evt, attr):
                v = getattr(evt, attr)
                if v:
                    self_us = float(v)
                    break

    # ---- inclusive device time ----
    incl_us = 0.0

    # 1) New, device-agnostic total
    for attr in ("device_time_total",):
        if hasattr(evt, attr):
            v = getattr(evt, attr)
            if v:
                incl_us = float(v)
                break

    # 2) CUDA-specific total
    if incl_us == 0.0:
        for attr in ("cuda_time_total",):
            if hasattr(evt, attr):
                v = getattr(evt, attr)
                if v:
                    incl_us = float(v)
                    break

    # 3) Per-call inclusive device time
    if incl_us == 0.0:
        for attr in ("device_time", "cuda_time"):
            if hasattr(evt, attr):
                v = getattr(evt, attr)
                if v:
                    incl_us = float(v)
                    break

    # If we still have no self time but we *do* have inclusive time,
    # approximate self by inclusive (better than zero for summaries).
    if self_us == 0.0 and incl_us > 0.0:
        self_us = incl_us

    return self_us, incl_us


# ----------------------- numeric summary -------------------------------------


def analyze_profiler(prof, rank: int) -> None:
    """
    Summarize compute vs comm vs overlap for a torch.profiler.Profile.

    Design choices:
    - We classify ops by name into two buckets:
        * compute_ops  ≈ matmuls, activation, etc.
        * comm_ops     ≈ NCCL collectives (all_gather, reduce_scatter, all_reduce).
    - For bucket totals we use **self device time** (self_device_time_total)
      when available. This avoids double-counting nested ops: each kernel's
      cost is charged only to the op that directly launched it.
    - If all device times are zero (CUPTI disabled / CPU run), we fall back to
      self CPU times so the summary still shows something meaningful.
    """

    events = prof.key_averages(group_by_input_shape=False)

    compute_ops = ("mm", "addmm", "matmul", "softmax", "gelu", "relu")
    comm_ops = ("all_gather", "reduce_scatter", "all_reduce")

    compute_dev_us = 0.0
    comm_dev_us = 0.0
    compute_cpu_us = 0.0
    comm_cpu_us = 0.0

    any_device_timing = False

    for evt in events:
        name = evt.key.lower()

        self_dev_us, _ = _get_device_times_us(evt)
        self_cpu_us = getattr(evt, "self_cpu_time_total", 0.0)

        if self_dev_us > 0.0:
            any_device_timing = True

        if any(op in name for op in compute_ops):
            compute_dev_us += self_dev_us
            compute_cpu_us += self_cpu_us
        elif any(op in name for op in comm_ops):
            comm_dev_us += self_dev_us
            comm_cpu_us += self_cpu_us

    print(f"\n=== PROFILER SUMMARY (rank={rank}) ===")

    # Prefer device timings; fall back to CPU if none are populated.
    if any_device_timing:
        compute_ms = compute_dev_us / 1000.0
        comm_ms = comm_dev_us / 1000.0
        source = "device_time (self_device_time_total)"
    else:
        compute_ms = compute_cpu_us / 1000.0
        comm_ms = comm_cpu_us / 1000.0
        source = "CPU (self_cpu_time_total)"
        if compute_ms == 0.0 and comm_ms == 0.0:
            print("No device or CPU self timings available; profiler likely empty.")
            print("=== END SUMMARY ===\n")
            return

    total_ms = compute_ms + comm_ms

    print(f"Timing source:           {source}")
    print(f"Compute time:            {compute_ms:8.2f} ms")
    print(f"Comm time (NCCL-ish):    {comm_ms:8.2f} ms")
    print(f"Total (compute+comm):    {total_ms:8.2f} ms")

    # Rough overlap estimate: how much of comm could be hidden under compute.
    hidden_comm = min(compute_ms, comm_ms)
    visible_comm = max(0.0, comm_ms - compute_ms)
    overlap_ratio = 100.0 * hidden_comm / comm_ms if comm_ms > 0.0 else 0.0

    print(f"\nHidden communication:    {hidden_comm:8.2f} ms")
    print(f"Visible communication:   {visible_comm:8.2f} ms")
    print(f"Overlap ratio:           {overlap_ratio:8.1f}%")

    # Top comm kernels by *self* device time
    print("\nBreakdown of top comm kernels:")
    comm_events = [e for e in events if any(op in e.key.lower() for op in comm_ops)]
    comm_events = sorted(
        comm_events,
        key=lambda e: _get_device_times_us(e)[0],  # self device time
        reverse=True,
    )

    for e in comm_events[:5]:
        self_dev_us, incl_dev_us = _get_device_times_us(e)
        self_ms = self_dev_us / 1000.0
        incl_ms = incl_dev_us / 1000.0
        print(
            f"  {e.key:30s} self={self_ms:8.2f} ms  "
            f"inclusive={incl_ms:8.2f} ms  calls={e.count}"
        )

    print("=== END SUMMARY ===\n")


# ----------------------- ASCII Gantt chart -----------------------------------


def print_ascii_gantt(prof, rank: int, width: int = 60) -> None:
    """
    ASCII Gantt-style timeline for compute vs comm vs overlap.

    We use the same classification as in `analyze_profiler`, and again rely
    on **self device time totals** (falling back to CPU if necessary).

    The chart is approximate – it doesn't reconstruct exact per-step overlap –
    but it gives a good *intuition* for how much of the step is compute-bound
    vs communication-bound.
    """

    events = prof.key_averages(group_by_input_shape=False)

    compute_ops = ("mm", "addmm", "matmul", "softmax", "gelu", "relu")
    comm_ops = ("all_gather", "reduce_scatter", "all_reduce")

    compute_dev_us = comm_dev_us = 0.0
    compute_cpu_us = comm_cpu_us = 0.0
    any_device_timing = False

    for e in events:
        name = e.key.lower()
        self_dev_us, _ = _get_device_times_us(e)
        self_cpu_us = getattr(e, "self_cpu_time_total", 0.0)

        if self_dev_us > 0.0:
            any_device_timing = True

        if any(op in name for op in compute_ops):
            compute_dev_us += self_dev_us
            compute_cpu_us += self_cpu_us
        elif any(op in name for op in comm_ops):
            comm_dev_us += self_dev_us
            comm_cpu_us += self_cpu_us

    # Choose timing source
    if any_device_timing:
        compute_ms = compute_dev_us / 1000.0
        comm_ms = comm_dev_us / 1000.0
        source = "device_time (self_device_time_total)"
    else:
        compute_ms = compute_cpu_us / 1000.0
        comm_ms = comm_cpu_us / 1000.0
        source = "CPU (self_cpu_time_total)"

    total_ms = compute_ms + comm_ms
    if total_ms <= 0.0:
        print(f"\n=== ASCII GANTT TIMELINE (rank={rank}) ===")
        print("No timing data available.")
        print("===========================================\n")
        return

    def scale(x: float) -> int:
        return max(0, min(width, int(width * (x / total_ms))))

    comp_bar = scale(compute_ms)
    overlap_ms = min(compute_ms, comm_ms)
    exposed_ms = max(0.0, comm_ms - compute_ms)
    overlap_bar = scale(overlap_ms)
    exposed_bar = scale(exposed_ms)

    compute_row = "█" * comp_bar
    comm_row = "▒" * overlap_bar + "░" * exposed_bar

    print(f"\n=== ASCII GANTT TIMELINE (rank={rank}) ===")
    print(f"Using {source}. Total step time ≈ {total_ms:.2f} ms\n")
    print(" Legend:")
    print("   █ = compute")
    print("   ▒ = overlapped comm (hidden by compute, approx.)")
    print("   ░ = exposed comm (stall)\n")

    print(" Compute:   " + compute_row)
    print(" NCCL-ish:  " + comm_row)

    overlap_ratio = (overlap_ms / comm_ms * 100.0) if comm_ms > 0.0 else 100.0
    print(f"\n Overlap ratio (approx): {overlap_ratio:.1f}%")
    print("===========================================\n")
