import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

# Colors matching YaFSDP style
COLOR_COMP = "#b7b7b7"  # compute background
COLOR_FWD = "#80a0ff"  # compute kernels (blue-ish)
COLOR_RS = "#70c070"  # reduce_scatter (green)
COLOR_AG = "#e070a0"  # all_gather (pink)
COLOR_ARROW = "#d03030"  # red

COMM_RS = ["reduce_scatter", "reducescatter"]
COMM_AG = ["all_gather", "allgather"]


def _load_events(path):
    with open(path) as f:
        trace = json.load(f)
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    return events


def _etype(evt_name):
    name = evt_name.lower()
    if any(k in name for k in COMM_RS):
        return "rs"
    if any(k in name for k in COMM_AG):
        return "ag"
    return "comp"


def _group_by_stream(events):
    """
    Returns: dict: stream_id → list of (type, start_ms, end_ms)
    """
    streams = defaultdict(list)

    for e in events:
        dur_us = e.get("dur", 0)
        if dur_us < 50:
            continue

        ts = e["ts"] / 1000.0
        dur = dur_us / 1000.0
        t = _etype(e["name"])

        stream = e.get("tid", 0)  # Chrome trace stores CUDA stream in 'tid'

        streams[stream].append((t, ts, ts + dur))

    # Normalize timeline so first op starts at 0
    t0 = min(min(s for _, s, _ in evts) for evts in streams.values())
    for s in streams:
        streams[s] = [(t, st - t0, en - t0) for t, st, en in streams[s]]

    return streams


def _draw_stream(ax, evts, label, total_ms):
    # Draw background bar
    ax.barh(0, total_ms, left=0, height=0.6, color=COLOR_COMP, edgecolor="none")

    for typ, s, e in evts:
        if typ == "comp":
            color = COLOR_FWD
        elif typ == "rs":
            color = COLOR_RS
        elif typ == "ag":
            color = COLOR_AG
        else:
            color = "gray"

        ax.barh(0, e - s, left=s, height=0.6, color=color)

    ax.set_xlim(0, total_ms)
    ax.set_yticks([])
    ax.set_title(label, fontsize=13)


def _pick_streams(streams):
    """
    Heuristic: Compute is always the *lowest-numbered CUDA stream*.
    NCCL comm streams have large tid.
    """
    stream_ids = sorted(streams.keys())

    compute = stream_ids[0]  # usually stream 7 or 13 for NCCL
    comm = [s for s in stream_ids[1:]]

    return compute, comm


def plot_comparisons(
    sync_dir="logs_sync", async_dir="logs_async", out="logs/overlap_comparison.png"
):
    # Load Chrome traces (rank0 only)
    sync_path = os.path.join(sync_dir, "trace_rank0.json")
    async_path = os.path.join(async_dir, "trace_rank0.json")

    sync_events = _load_events(sync_path)
    async_events = _load_events(async_path)

    sync_streams = _group_by_stream(sync_events)
    async_streams = _group_by_stream(async_events)

    sync_compute, sync_comm = _pick_streams(sync_streams)
    async_compute, async_comm = _pick_streams(async_streams)

    # Compute total duration
    sync_total = max(en for evts in sync_streams.values() for _, _, en in evts)
    async_total = max(en for evts in async_streams.values() for _, _, en in evts)

    fig, axs = plt.subplots(6, 1, figsize=(16, 10), sharex=False)

    # Async (top 3 rows)
    _draw_stream(axs[0], async_streams[async_compute], "ASYNC: Compute Stream", async_total)
    _draw_stream(axs[1], async_streams.get(async_comm[0], []), "ASYNC: NCCL RS Stream", async_total)
    if len(async_comm) > 1:
        _draw_stream(
            axs[2], async_streams.get(async_comm[1], []), "ASYNC: NCCL AG Stream", async_total
        )

    # Sync (bottom 3 rows)
    _draw_stream(axs[3], sync_streams[sync_compute], "SYNC: Compute Stream", sync_total)
    _draw_stream(axs[4], sync_streams.get(sync_comm[0], []), "SYNC: NCCL RS Stream", sync_total)
    if len(sync_comm) > 1:
        _draw_stream(axs[5], sync_streams.get(sync_comm[1], []), "SYNC: NCCL AG Stream", sync_total)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved YaFSDP-style comparison → {out}")


__all__ = ["plot_comparisons"]
