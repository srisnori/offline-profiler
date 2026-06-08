# Offline LLM Profiler

Tool to benchmark **LLM compute performance** and ** communication cost** for models.

---
- Benchmarks:
  - **MLP** 
  - **Multi-Head Attention**
  - **Grouped Query Attention**

- Measures:
  - CPU compute time
  - GPU compute time (working on it)
  - Network latency
  - Bandwidth
  - Communication cost across nodes

- Uses HuggingFace model to derive configs:
  - hidden size
  - number of heads
  - model architecture

---

## Inputs
- **Model** 
- **Batch size**
- **Sequence length**
- **Attention mechanism** (`MHA`, `GQA`, `MLP`)
- **GPU type**
- **GPU memory (GB)**
- **IP addresses**

---

## Outputs

- **t_attn^CPU** → Attention compute time (CPU)
- **t_attn^GPU** → Attention compute time (GPU)
- **t_mlp^CPU / t_mlp^GPU** → MLP compute time
- **Network latency**
- **Bandwidth**
- **T_comm** → Communication time

---

## Run

```bash
git clone https://github.com/srisnori/offline-profiler.git
cd offline-profiler

python -m venv venv
source venv/bin/activate

python profiler.py
