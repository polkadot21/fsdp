import glob
import json
import os

import matplotlib.pyplot as plt

# Visual spec matching YaFSDP
COLOR_COMP = "#d7d7d7"  # light gray compute bar
COLOR_RS = "#70c070"  # green reduce_scatter
COLOR_AG = "#e070a0"  # pink all_gather
COLOR_ARROW = "#d03030"  # red arrows

COMM_RS = ["reduce_scatter"]
COMM_AG = ["all_gather", "allreduce"]


def _load_events(path):
    with open(path) as f:
        tr = json.load(f)
    return [e for e in tr["traceEvents"] if e.get("ph") == "X"]


def _etype(evt):
    name = evt["name"].lower()

    if any(k in name for k in COMM_RS):
        return "rs"
    if any(k in name for k in COMM_AG):
        return "ag"
    return "comp"


def _intervals(events):
    ops = []
    for e in events:
        dur_us = e.get("dur", 0)
        if dur_us <= 80:  # ignore tiny ops, reduces noise
            continue

        ts = e["ts"] / 1000.0
        dur = dur_us / 1000.0
        t = _etype(e)
        ops.append((t, ts, ts + dur))

    if not ops:
        return [], 0.0

    # Normalize timeline so first compute starts at 0 (as in YaFSDP)
    t0 = min(s for _, s, _ in ops)
    ops = [(t, s - t0, e - t0) for t, s, e in ops]

    total = max(e for _, _, e in ops)
    return ops, total


def _draw_row(ax, ops, total_ms, label):
    # Draw compute backing region
    ax.barh(0, total_ms, left=0, height=0.6, color=COLOR_COMP, edgecolor="none")

    for typ, s, e in ops:
        if typ == "rs":
            ax.barh(0, e - s, left=s, height=0.6, color=COLOR_RS)
        elif typ == "ag":
            ax.barh(0, e - s, left=s, height=0.6, color=COLOR_AG)

    # Draw red arrow showing full step width
    ax.annotate(
        "",
        xy=(0, 0.85),
        xytext=(total_ms, 0.85),
        arrowprops=dict(arrowstyle="<->", color=COLOR_ARROW, lw=2),
        ha="center",
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


def plot_comparisons(
    sync_dir="logs_sync", async_dir="logs_async", out="logs/overlap_comparison.png"
):
    # Locate trace files
    sync_f = _first(glob.glob(os.path.join(sync_dir, "trace_rank0.json")))
    async_f = _first(glob.glob(os.path.join(async_dir, "trace_rank0.json")))

    if sync_f is None:
        raise FileNotFoundError("SYNC trace not found at logs_sync/trace_rank0.json")
    if async_f is None:
        raise FileNotFoundError("ASYNC trace not found at logs_async/trace_rank0.json")

    sync_ops, sync_total = _intervals(_load_events(sync_f))
    async_ops, async_total = _intervals(_load_events(async_f))

    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

    _draw_row(axs[0], async_ops, async_total, "YaFSDP (ASYNC overlapped)")
    _draw_row(axs[1], sync_ops, sync_total, "FSDP baseline (SYNC)")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved YaFSDP-style comparison figure → {out}")


def _first(lst):
    return lst[0] if lst else None


__all__ = ["plot_comparisons"]
