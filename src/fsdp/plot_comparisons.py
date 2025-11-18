import glob
import json
import os

import matplotlib.pyplot as plt

# -------- Colors (match YaFSDP) --------
COLOR_COMP_BG = "#d7d7d7"  # light gray background for compute lane
COLOR_COMP = "#6495ed"  # blue-ish compute kernel foreground
COLOR_RS = "#70c070"  # green reduce_scatter
COLOR_AG = "#e070a0"  # pink all_gather
COLOR_AR = "#60c0c0"  # teal all_reduce (rare)
COLOR_ARROW = "#d03030"  # red total-step arrow


# ------------------------------
# Compute + NCCL classification
# ------------------------------

COMPUTE_KEYWORDS = [
    "gemm",
    "sgemm",
    "hgemm",
    "dgemm",
    "cublas",
    "matmul",
    "flash",
    "attn",
    "attention",
    "ampere",  # ampere_fma / ampere_mma -> fused LN/MLP kernels
    "relu",
    "gelu",
    "silu",
    "layer_norm",
    "softmax",
    "fused",
    "bias",
    "add",
    "mul",
]


def is_compute_kernel(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in COMPUTE_KEYWORDS)


def comm_type(name: str):
    n = name.lower()

    if "reduce_scatter" in n:
        return "rs"
    if "all_gather" in n:
        return "ag"
    if "all_reduce" in n:
        return "ar"

    # other nccl kernels (Copy, Kernel_Sum, LL, etc.) should count as comm
    if "nccl" in n:
        return "misc"

    return None


# ------------------------------
# Event loading + grouping
# ------------------------------


def load_events(path):
    with open(path) as f:
        tr = json.load(f)
    return [e for e in tr["traceEvents"] if e.get("ph") == "X"]


def group_by_stream(events):
    """
    Return {tid: [events]} for CUDA streams only.
    In Chrome trace, CUDA kernel events normally have a "cat": "Kernel"
    or "cat": "cudaLaunchKernel".
    """
    streams = {}
    for e in events:
        cat = e.get("cat", "")
        if "Kernel" not in cat and "cuda" not in cat.lower():
            continue
        tid = e.get("tid", None)
        if tid is None:
            continue
        streams.setdefault(tid, []).append(e)
    return streams


def pick_streams(streams):
    """
    Heuristic to pick:
      - compute_stream: stream whose events contain majority compute kernels
      - comm_stream: stream whose events contain majority nccl kernels
    """
    best_compute = None
    best_comm = None
    compute_score = -1
    comm_score = -1

    for tid, evts in streams.items():
        comp = 0
        comm = 0
        for e in evts:
            name = e["name"]
            if is_compute_kernel(name):
                comp += e.get("dur", 0)
            elif comm_type(name) is not None:
                comm += e.get("dur", 0)

        if comp > compute_score:
            compute_score = comp
            best_compute = tid
        if comm > comm_score:
            comm_score = comm
            best_comm = tid

    return best_compute, best_comm


def extract_intervals(events, stream_id):
    """
    Extract (type, start_ms, end_ms) for events in the chosen stream.
    """
    ops = []
    for e in events:
        if e.get("tid") != stream_id:
            continue

        dur_us = e.get("dur", 0)
        if dur_us <= 50:  # ignore tiny noise kernels
            continue

        ts = e["ts"] / 1000.0
        ed = ts + dur_us / 1000.0

        name = e["name"]
        ctype = comm_type(name)
        if ctype is not None:
            # classify comm
            if ctype == "rs":
                ops.append(("rs", ts, ed))
            elif ctype == "ag":
                ops.append(("ag", ts, ed))
            elif ctype == "ar":
                ops.append(("ar", ts, ed))
            else:
                # misc nccl -> treat as comm too
                ops.append(("nccl", ts, ed))
        else:
            # compute or ignore
            if is_compute_kernel(name):
                ops.append(("comp", ts, ed))

    if not ops:
        return [], 0.0

    # Normalize start to 0
    t0 = min(s for _, s, _ in ops)
    ops = [(t, s - t0, e - t0) for (t, s, e) in ops]
    total = max(e for _, _, e in ops)
    return ops, total


# ------------------------------
# Plotting
# ------------------------------


def draw_row(ax, comp_ops, comm_ops, total_ms, label):
    """
    Draw 2-lane row like YaFSDP:
      Lane 0: compute
      Lane 1: comm (RS + AG)
    """
    height = 0.6

    # --- Compute lane background ---
    ax.barh(1, total_ms, left=0, height=height, color=COLOR_COMP_BG, edgecolor="none")

    # Draw compute kernels
    for typ, s, e in comp_ops:
        if typ == "comp":
            ax.barh(1, e - s, left=s, height=height, color=COLOR_COMP)

    # --- Communication lane ---
    for typ, s, e in comm_ops:
        if typ == "rs":
            ax.barh(0, e - s, left=s, height=height, color=COLOR_RS)
        elif typ == "ag":
            ax.barh(0, e - s, left=s, height=height, color=COLOR_AG)
        elif typ == "ar":
            ax.barh(0, e - s, left=s, height=height, color=COLOR_AR)
        elif typ == "nccl":
            ax.barh(0, e - s, left=s, height=height, color=COLOR_AR)

    # --- Red arrow for total step ---
    ax.annotate(
        "",
        xy=(0, 1.7),
        xytext=(total_ms, 1.7),
        arrowprops=dict(arrowstyle="<->", color=COLOR_ARROW, lw=2),
    )
    ax.text(
        total_ms / 2,
        1.85,
        f"{total_ms:.0f} ms",
        color=COLOR_ARROW,
        fontsize=12,
        ha="center",
        va="bottom",
    )

    ax.set_xlim(0, total_ms)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Compute", "Comm"])
    ax.set_title(label, fontsize=14, pad=15)
    ax.set_xlabel("Time (ms)")


# ------------------------------
# Main entry
# ------------------------------


def plot_comparisons(
    sync_dir="logs_sync", async_dir="logs_async", out="logs/overlap_comparison.png"
):
    sync_f = _first(glob.glob(os.path.join(sync_dir, "trace_rank0.json")))
    async_f = _first(glob.glob(os.path.join(async_dir, "trace_rank0.json")))

    if sync_f is None:
        raise FileNotFoundError("SYNC trace not found")
    if async_f is None:
        raise FileNotFoundError("ASYNC trace not found")

    sync_events = load_events(sync_f)
    async_events = load_events(async_f)

    sync_streams = group_by_stream(sync_events)
    async_streams = group_by_stream(async_events)

    sync_comp_tid, sync_comm_tid = pick_streams(sync_streams)
    async_comp_tid, async_comm_tid = pick_streams(async_streams)

    sync_comp_ops, sync_total = extract_intervals(sync_events, sync_comp_tid)
    sync_comm_ops, _ = extract_intervals(sync_events, sync_comm_tid)

    async_comp_ops, async_total = extract_intervals(async_events, async_comp_tid)
    async_comm_ops, _ = extract_intervals(async_events, async_comm_tid)

    fig, axs = plt.subplots(2, 1, figsize=(16, 8))

    draw_row(axs[0], async_comp_ops, async_comm_ops, async_total, "YaFSDP-like ASYNC (overlapped)")

    draw_row(axs[1], sync_comp_ops, sync_comm_ops, sync_total, "FSDP SYNC baseline")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved comparison figure → {out}")


def _first(lst):
    return lst[0] if lst else None


__all__ = ["plot_comparisons"]
