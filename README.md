# Offline Profiler

## Key Features

- **Device Diagnostics:** Automatic detection of execution substrate (`CUDA` / `CPU`) with GPU memory auto-allocation.
- **Kernel Benchmarking:** Measures per-layer execution times for key transformer building blocks:
  - Multi-Head Attention (**MHA**)
  - Grouped-Query Attention (**GQA**)
  - Multi-Layer Perceptron (**MLP**)
- **Network Matrix Profiling:**
  - Inter-node latency ($d_{i \to j}$) and bandwidth ($B_{i \to j}$).
- **Evaluation Environment Presets ($E_1$–$E_6$):** Built-in support for BloomBee evaluation environments:
  - **E1–E5:** Controlled homogeneous setups ($45\text{ Gbps}$ cluster down to $20\text{ Mbps}$).
  - **E6:** Measured heterogeneous WAN matrix across California, New Jersey, and Canada.
- **Dynamic Programming Placement Solver (`dp_scheduler`):** Automatically computes the optimal pipeline layer split per node (e.g., `[13, 13, 14]`) to minimize end-to-end execution latency.

---

## Supported Inputs

When running `profiler.py`, you will be prompted for:

| Input | Description | Default |
| :--- | :--- | :--- |
| **Environment** | BloomBee preset (`E1`–`E6`) or `0` for Custom IP probing | `0` (Custom) |
| **Model Config** | Model name, Batch Size, Sequence Length | `llama`, `32`, `128` |
| **Architecture** | Num Layers, Num Heads, Embed Dim | `40`, `52`, `6656` |
| **Attention Kernel** | `mha`, `gqa`, or `mlp` | `mha` |
| **VRAM Capacity** | GPU Memory per node in GB | Device Total VRAM |
| **Node Cluster** | Distributed IP addresses or node identifiers | Prompted or preset |

---

## Profiler Outputs

- **Compute Metrics:**
  - $t_{\text{attn}}^{\text{CPU}} / t_{\text{attn}}^{\text{GPU}}$: Single-layer Attention execution time.
  - $t_{\text{mlp}}^{\text{CPU}} / t_{\text{mlp}}^{\text{GPU}}$: Single-layer MLP execution time.
  - $T_{\text{layer}}$: Total single-layer compute latency ($t_{\text{attn}} + t_{\text{mlp}}$).
- **Network Topology Metrics:**
  - Inter-Node Latency ($s$) and Link Bandwidth ($\text{B/s}$).
  - $T_{\text{comm}}$: Activation tensor payload transfer overhead per pipeline stage.
- **Optimal Pipeline Assignment:**
  - **Layer Assignment:** Array containing assigned layer counts per node (e.g., `[13, 13, 14]`).
  - **Total Cost (DP):** Minimum theoretical execution time per pass ($s$).

---

## Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/srisnori/offline-profiler.git
cd offline-profiler

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch numpy
