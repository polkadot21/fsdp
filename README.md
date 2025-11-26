
# FSDP with YaFSDP-Style Asynchronous Overlap

### **📌 Summary**
This repository implements a **manual Fully Sharded Data Parallel (FSDP)** training loop from scratch. It explicitly manages CUDA streams to achieve **perfect computation-communication overlap**, following the architectural principles of **YaFSDP**.

Unlike standard FSDP wrappers, this implementation exposes the raw mechanics of distributed training, offering a "clean room" environment to study:
- **Bidirectional Pipelining:** Fetching the *next* layer's weights while computing the *current* layer.
- **Memory Safety:** Solving complex PyTorch Autograd lifecycle issues (Tensor ID reuse collisions and stale pointers).
- **Graph Topology Control:** Using `GateGradFlow` to enforce strict ordering between gradient calculation and reduction, preventing race conditions.

---

## **🎯 Architecture & Key Features**

This implementation goes beyond simple async handles. It builds a custom engine to handle the delicate dance between the **Compute Stream** and the **Communication Stream**.

| Feature | Implementation Detail |
|--------|-----------------------|
| **Static Memory Pool** | Fixed-size "Ping-Pong" buffers (Pre-allocated). Zero memory fragmentation/reallocation during training. |
| **GateGradFlow** | A custom `autograd.Function` "fence" that guarantees weight gradients are fully computed *before* `ReduceScatter` launches. |
| **Storage Rescue** | Advanced Autograd hooks that survive Python Garbage Collection and Tensor ID reuse (the "Stale Pointer" bugs). |
| **Stream Pipelining** | Explicit `cuda.Event` synchronization to ensure Compute never waits for Communication (unless bandwidth bound). |
| **Zero-Copy** | Parameters are materialized as *views* into the global buffer, avoiding expensive `memcpy`. |
| **Pre-Backward Trigger**| A `register_full_backward_pre_hook` that launches the `AllGather` for layer `N-1` immediately when layer `N` starts its backward pass. |

---

## **🧪 Experiment Setup**

### **Hardware**
- Minimum 2× NVIDIA GPUs (Tested on H100 80GB)
- Interconnect: NVLink or high-speed TCP/IB

### **Software**
- **PyTorch Distributed** (NCCL Backend)
- **Chrome Tracing** (via `torch.profiler`)
- **UV** (Modern Python dependency management)

---

## **📂 Execution**

The entry point is flexible, allowing you to run a fast "Proof of Concept" (PoC) to verify overlap, or a massive "Giant" simulation to stress test memory.

### **Python / Jupyter API**

```python
from fsdp import run_on_cloud

# Run the fast verification (Checks overlap & convergence)
run_on_cloud(mode="poc")

# Run the heavy simulation (Checks OOM safety & massive throughput)
# run_on_cloud(mode="giant")
```

### CLI / Makefile

```bash
make test-real-model
```

### Profiling Results

![Async profiling](assets/async_profiling.png)

The trace above demonstrates Perfect Compute Utilization using `Overlap: True`.

- Top Row (Compute Stream).

- Bottom Row (Comm Stream): AllGather (Forward/Backward) and ReduceScatter happen entirely in the background.


## **📚 Sources & References**
- **YaFSDP**
- PyTorch FSDP RFCs
- DeepSpeed ZeRO Stage-3 implementation notes
- NVIDIA NCCL collective performance docs
