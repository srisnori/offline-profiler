import ipaddress
import statistics
import torch

from bandwidth import ENVIRONMENTS, get_bandwidth
from communication_time import communication_time
from gqa_benchmark import GQA_CPU, GQA_GPU
from mha_benchmark import MHA_CPU, MHA_GPU
from mlp_benchmark import MLP_CPU, MLP_GPU
from network_graph import network_graph
from scheduler import dp_scheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

# Environment Selection 
print("\n--- Environment Selection ---")
print("[0] Custom IPs")
for key, data in ENVIRONMENTS.items():
    print(f"[{key}] {data['description']}")

env_choice = input("\nSelect Environment (E1-E6 or 0 for Custom, default '0'): ").strip().upper()

use_preset = env_choice in ENVIRONMENTS
selected_env = env_choice if use_preset else None

if selected_env == "E6":
    nodes = ["California", "New Jersey", "Canada"]
    print(f"[Selected Preset] E6 Heterogeneous WAN -> Nodes: {nodes}")
elif use_preset:
    nodes = ["Node_1", "Node_2", "Node_3"]
    print(f"[Selected Preset] {selected_env} ({ENVIRONMENTS[selected_env]['description']}) -> Nodes: {nodes}")
else:
    print("[Mode] Custom IP Profiling selected.")
    raw_ips = input("Enter Distributed IPs (comma-separated): ").strip()
    nodes = [ip.strip() for ip in raw_ips.split(",") if ip.strip() and not ipaddress.IPv4Address(ip.strip()).is_unspecified]
    if not nodes:
        raise ValueError("No valid IP addresses provided. Exiting.")

# Model Inputs 
model_name = input("Model (default 'llama'): ").strip() or "llama"
batch_size = int(input("Batch Size (default 32): ") or 32)
seq_len = int(input("Seq Length (default 128): ") or 128)
num_layers = int(input("Num Layers (default 40): ") or 40)
num_heads = int(input("Num Heads (default 52): ") or 52)
embed_dim = int(input("Embed Dim (default 6656): ") or 6656)
attention_mechanism = input("Attention (MHA/GQA/MLP, default 'mha'): ").strip().lower() or "mha"

if embed_dim % num_heads != 0:
    raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

num_trials = int(input("Number of Benchmark Trials to Average (default 20): ") or 20)

# Configure Uniform GPU VRAM and GPU Counts Per Node
default_vram = int(torch.cuda.get_device_properties(0).total_memory / (1024**3)) if device == "cuda" else 40

print("\n--- Configure GPU Hardware per Node ---")
node_gpu_counts = []
node_gpu_vrams = []

for node in nodes:
    print(f"\n[{node}]")
    count_input = input(f"  Number of GPUs (default 1): ").strip()
    num_g = int(count_input) if count_input else 1
    
    vram_input = input(f"  VRAM per GPU in GB (default {default_vram}): ").strip()
    vram_g = float(vram_input) if vram_input else float(default_vram)
    
    total_mem = num_g * vram_g
    node_gpu_counts.append(num_g)
    node_gpu_vrams.append(vram_g)
    print(f"  -> Total: {num_g} GPU(s) x {vram_g:.0f} GB = {total_mem:.1f} GB VRAM")




# Compute Benchmarks (Averaged over N trials)
print(f"\n--- Running Compute Benchmarks over {num_trials} Trials ({device.upper()}) ---")

mlp_cpu_runs, mlp_gpu_runs = [], []
attn_cpu_runs, attn_gpu_runs = [], []

for t in range(num_trials):
    # MLP
    m_cpu = MLP_CPU(embed_dim, batch_size, seq_len)
    m_gpu = MLP_GPU(embed_dim, batch_size, seq_len) if device == "cuda" else m_cpu
    mlp_cpu_runs.append(m_cpu)
    mlp_gpu_runs.append(m_gpu)

    # Attention
    if attention_mechanism == "mha":
        mech = "MHA"
        a_cpu = MHA_CPU(embed_dim, batch_size, seq_len, num_heads)
        a_gpu = MHA_GPU(embed_dim, batch_size, seq_len, num_heads) if device == "cuda" else a_cpu
    elif attention_mechanism == "gqa":
        mech = "GQA"
        a_cpu = GQA_CPU(embed_dim, batch_size, seq_len, num_heads)
        a_gpu = GQA_GPU(embed_dim, batch_size, seq_len, num_heads) if device == "cuda" else a_cpu
    elif attention_mechanism == "mlp":
        mech = "MLP"
        a_cpu, a_gpu = m_cpu, m_gpu
    else:
        raise ValueError("Attention mechanism must be mha, gqa, or mlp.")

    attn_cpu_runs.append(a_cpu)
    attn_gpu_runs.append(a_gpu)

mlp_cpu = statistics.mean(mlp_cpu_runs)
mlp_gpu = statistics.mean(mlp_gpu_runs)
attn_cpu = statistics.mean(attn_cpu_runs)
attn_gpu = statistics.mean(attn_gpu_runs)

mlp_gpu_std = statistics.stdev(mlp_gpu_runs) if len(mlp_gpu_runs) > 1 else 0.0
attn_gpu_std = statistics.stdev(attn_gpu_runs) if len(attn_gpu_runs) > 1 else 0.0

print(f"MLP Time (CPU Mean): {mlp_cpu:.4f} s | MLP Time (GPU Mean): {mlp_gpu:.4f} s (± {mlp_gpu_std:.4f} s)")
print(f"{mech} Time (CPU Mean): {attn_cpu:.4f} s | {mech} Time (GPU Mean): {attn_gpu:.4f} s (± {attn_gpu_std:.4f} s)")

layer_compute_gpu = attn_gpu + mlp_gpu
print(f"Single Layer GPU Time Mean ({mech} + MLP): {layer_compute_gpu:.4f} s")
print(f"Total Model GPU Time ({num_layers} Layers): {(layer_compute_gpu * num_layers):.4f} s")

# Network Matrix Profiling 
print("\n--- Network Matrix Profiling ---")
graph = {a: {} for a in nodes}

if len(nodes) < 2:
    print("[Warning] Need at least 2 nodes for inter-node profiling.")
    edge_latencies = [0.0]
    edge_bandwidths = [1_250_000_000.0]
else:
    if use_preset:
        for s in nodes:
            for r in nodes:
                if s == r:
                    continue
                bw = get_bandwidth(sender=s, receiver=r, env=selected_env)
                lat = 0.050 if selected_env == "E6" else 0.003
                t_comm = communication_time(lat, bw, batch_size, seq_len, embed_dim)
                graph[s][r] = {"latency": lat, "bandwidth": bw, "t_comm": t_comm}
                print(f"[{s} -> {r}] Latency: {lat:.4f} s | Bandwidth: {bw:.2f} B/s | T_comm: {t_comm:.4f} s")
    else:
        try:
            graph = network_graph(nodes)
        except Exception as e:
            print(f"Network probing failed ({e}). Falling back to default link estimation.")
            graph = {
                a: {b: {"latency": 0.003, "bandwidth": 1_250_000_000.0} for b in nodes if b != a}
                for a in nodes
            }

        for s in graph:
            for r in graph[s]:
                lat = graph[s][r].get("latency", 0.003)
                bw = graph[s][r].get("bandwidth", 1_250_000_000.0)
                t_comm = communication_time(lat, bw, batch_size, seq_len, embed_dim)
                graph[s][r]["t_comm"] = t_comm
                print(f"[{s} -> {r}] Latency: {lat:.4f} s | Bandwidth: {bw:.2f} B/s | T_comm: {t_comm:.4f} s")

    # Extract sequential pipeline edges: Node 0 -> Node 1, Node 1 -> Node 2, ...
    edge_latencies = []
    edge_bandwidths = []
    for i in range(len(nodes) - 1):
        s, r = nodes[i], nodes[i + 1]
        lat = graph[s].get(r, {}).get("latency", 0.003)
        bw = graph[s].get(r, {}).get("bandwidth", 1_250_000_000.0)
        edge_latencies.append(lat)
        edge_bandwidths.append(bw)

# Dynamic Programming Scheduler 
print("\n--- Dynamic Programming Scheduler ---")
assignment, total_cost = dp_scheduler(
    numLayers=num_layers,
    numNodes=len(nodes),
    t_mlp=mlp_gpu,
    t_attn_gpu=attn_gpu,
    t_attn_cpu=attn_cpu,
    latency=edge_latencies,
    bandwidth=edge_bandwidths,
    batchSize=batch_size,
    seqLen=seq_len,
    embedDim=embed_dim,
    num_gpus=node_gpu_counts,   
    gpu_vrams=node_gpu_vrams,  
    minGpuMem=0.0,
)

print(f"\nEnvironment Mode: {selected_env if selected_env else 'Custom IPs'}")
print(f"Layer Assignment per Node: {assignment}")
print(f"Bottleneck Stage Cost (DP): {total_cost:.4f}s")