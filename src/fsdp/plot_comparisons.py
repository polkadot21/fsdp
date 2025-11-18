import glob
import json
import os

import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Color palette (YaFSDP-inspired)
# -------------------------------------------------------------
COLOR_BG = "#e8e8e8"  # background compute lane (light grey)
COLOR_COMP = "#7aa7ff"  # compute kernels (blue-ish)
COLOR_RS = "#70c070"  # reduce_scatter (green)
COLOR_AG = "#e070a0"  # all_gather (pink)
COLOR_AR = "#60c0d0"  # all_reduce or other NCCL (teal)
COLOR_ARROW = "#d03030"  # red step arrow text & line


# -------------------------------------------------------------
# Utility to load profiler trace
# -------------------------------------------------------------
def _load_events(path):
    with open(path) as f:
        tr = json.load(f)
    return [e for e in tr["traceEvents"] if e.get("ph") == "X"]


# -------------------------------------------------------------
# Classify CUDA kernels into logical lanes
# -------------------------------------------------------------
def _etype(evt):
    name = evt["name"].lower()

    # NCCL kernels
    if "reduce_scatter" in name or "reducescatter" in name:
        return "rs"
    if "all_gather" in name or "allgather" in name:
        return "ag"
    if "all_reduce" in name or "allreduce" in name:
        return "ar"
    if "nccl" in name:
        return "ar"  # all other NCCL into generic comm

    # Everything non-NCCL is compute
    return "comp"


# -------------------------------------------------------------
# Convert raw events → (t0, t1) intervals in ms
# -------------------------------------------------------------
def _to_intervals(events, min_us=50):
    ops = []

    for e in events:
        dur_us = e.get("dur", 0)
        if dur_us < min_us:
            continue
        ts = e["ts"] / 1000.0
        dur = dur_us / 1000.0
        ops.append((_etype(e), ts, ts + dur))

    if not ops:
        return [], 0.0

    # Normalize timeline so that the first event starts at 0
    t0 = min(s for _, s, _ in ops)
    ops = [(t, s - t0, e - t0) for t, s, e in ops]
    total = max(e for _, _, e in ops)
    return ops, total


# -------------------------------------------------------------
# Draw a single row (compute or comm)
# -------------------------------------------------------------
def _draw_row(ax, ops, total_ms, label):
    # background
    ax.barh(0, total_ms, left=0, height=0.6, color=COLOR_BG, edgecolor="none")

    # draw kernels
    for typ, s, e in ops:
        if typ == "comp":
            ax.barh(0, e - s, left=s, height=0.6, color=COLOR_COMP, edgecolor="none")
        elif typ == "rs":
            ax.barh(0, e - s, left=s, height=0.6, color=COLOR_RS, edgecolor="none")
        elif typ == "ag":
            ax.barh(0, e - s, left=s, height=0.6, color=COLOR_AG, edgecolor="none")
        elif typ == "ar":
            ax.barh(0, e - s, left=s, height=0.6, color=COLOR_AR, edgecolor="none")

    # Step duration arrow
    ax.annotate(
        "",
        xy=(0, 0.85),
        xytext=(total_ms, 0.85),
        arrowprops=dict(arrowstyle="<->", color=COLOR_ARROW, lw=2),
    )
    ax.text(
        total_ms / 2,
        0.95,
        f"{total_ms:.0f} ms",
        color=COLOR_ARROW,
        fontsize=12,
        ha="center",
        va="bottom",
    )

    ax.set_xlim(0, total_ms)
    ax.set_yticks([])
    ax.set_title(label, fontsize=13, pad=15)
    ax.set_xlabel("Time (ms)")


# -------------------------------------------------------------
# Split events into:
#   - compute ops
#   - comm ops (rs/ag/ar) combined
# -------------------------------------------------------------
def _split_compute_and_comm(ops):
    comp = [(t, s, e) for (t, s, e) in ops if t == "comp"]
    comm = [(t, s, e) for (t, s, e) in ops if t != "comp"]
    return comp, comm


# -------------------------------------------------------------
# Main public API
# -------------------------------------------------------------
def plot_comparisons(
    sync_dir="logs_sync",
    async_dir="logs_async",
    out="logs/overlap_comparison.png",
):
    # --------------------------------------------------------
    # Load traces (rank0 only)
    # --------------------------------------------------------
    sync_path = _first(glob.glob(os.path.join(sync_dir, "trace_rank0.json")))
    async_path = _first(glob.glob(os.path.join(async_dir, "trace_rank0.json")))

    if sync_path is None:
        raise FileNotFoundError("SYNC trace not found at logs_sync/trace_rank0.json")
    if async_path is None:
        raise FileNotFoundError("ASYNC trace not found at logs_async/trace_rank0.json")

    sync_ops_raw = _load_events(sync_path)
    async_ops_raw = _load_events(async_path)

    sync_ops, sync_total = _to_intervals(sync_ops_raw)
    async_ops, async_total = _to_intervals(async_ops_raw)

    sync_comp, sync_comm = _split_compute_and_comm(sync_ops)
    async_comp, async_comm = _split_compute_and_comm(async_ops)

    # --------------------------------------------------------
    # Create figure like YaFSDP:
    # Row1: Compute
    # Row2: Communication (RS + AG + AR)
    # --------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

    # ASYNC
    _draw_row(axs[0], async_comp + async_comm, async_total, "YaFSDP-style ASYNC (overlapped)")

    # SYNC
    _draw_row(axs[1], sync_comp + sync_comm, sync_total, "FSDP baseline SYNC")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved YaFSDP-style comparison figure → {out}")


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _first(lst):
    return lst[0] if lst else None


__all__ = ["plot_comparisons"]
