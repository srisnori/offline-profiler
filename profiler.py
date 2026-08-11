import ipaddress
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

# Env Selection 
print("\n--- Environment Selection ---")
print("[0] Custom IPs")
for key, data in ENVIRONMENTS.items():
    print(f"[{key}] {data['description']}")

env_choice = (input("\nSelect Environment (E1-E6 or 0 for Custom, default '0'): ").strip().upper())

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
    nodes = [
        ip.strip()
        for ip in raw_ips.split(",")
        if ip.strip() and not ipaddress.IPv4Address(ip.strip()).is_unspecified
    ]
    if not nodes:
        raise ValueError("No valid IP addresses provided. Exiting.")

# Model Inputs 
model_name = input("Model (default 'llama'): ").strip() or "llama"
batch_size = int(input("Batch Size (default 32): ") or 32)
seq_len = int(input("Seq Length (default 128): ") or 128)
num_layers = int(input("Num Layers (default 40): ") or 40)
num_heads = int(input("Num Heads (default 52): ") or 52)
embed_dim = int(input("Embed Dim (default 6656): ") or 6656)
attention_mechanism = (input("Attention (MHA/GQA/MLP, default 'mha'): ").strip().lower() or "mha")

if device == "cuda":
    default_vram = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
    vram_in = input(f"GPU Memory GB (default {default_vram}): ").strip()
    gpu_mem = int(vram_in) if vram_in else default_vram
else:
    gpu_mem = 48

# Benchmarks 
print(f"\n--- Running Compute Benchmarks ({device.upper()}) ---")

mlp_cpu = MLP_CPU(embed_dim, batch_size, seq_len)
mlp_gpu = (MLP_GPU(embed_dim, batch_size, seq_len) if device == "cuda" else mlp_cpu)

if attention_mechanism == "mha":
    mech = "MHA"
    attn_cpu = MHA_CPU(embed_dim, batch_size, seq_len, num_heads)
    attn_gpu = (MHA_GPU(embed_dim, batch_size, seq_len, num_heads)
        if device == "cuda"
        else attn_cpu
    )
elif attention_mechanism == "gqa":
    mech = "GQA"
    attn_cpu = GQA_CPU(embed_dim, batch_size, seq_len, num_heads)
    attn_gpu = (
        GQA_GPU(embed_dim, batch_size, seq_len, num_heads)
        if device == "cuda"
        else attn_cpu
    )
elif attention_mechanism == "mlp":
    mech = "MLP"
    attn_cpu = mlp_cpu
    attn_gpu = mlp_gpu
else:
    raise ValueError("Attention mechanism must be mha, gqa, or mlp.")

print(f"MLP Time (CPU): {mlp_cpu:.4f} s | MLP Time (GPU): {mlp_gpu:.4f} s")
print(f"{mech} Time (CPU): {attn_cpu:.4f} s | {mech} Time (GPU): {attn_gpu:.4f} s")

layer_compute_gpu = attn_gpu + mlp_gpu
print(f"Single Layer GPU Time ({mech} + MLP): {layer_compute_gpu:.4f} s")
print(f"Total Model GPU Time ({num_layers} Layers): {(layer_compute_gpu * num_layers):.4f} s")

# Network Matrix Profiling 
print("\n--- Network Matrix Profiling ---")
graph = {a: {} for a in nodes}

if len(nodes) < 2:
    print("[Warning] Need at least 2 nodes for inter-node profiling.")
    lat_ab, bw_ab, t_comm_ab = 0.0, 1_250_000_000.0, 0.0
else:
    if use_preset:
        for s in nodes:
            for r in nodes:
                if s == r:
                    continue
                bw = get_bandwidth(sender=s, receiver=r, env=selected_env)
                lat = 0.050 if selected_env == "E6" else 0.003
                t_comm = communication_time(
                    lat, bw, batch_size, seq_len, embed_dim
                )
                graph[s][r] = {"latency": lat, "bandwidth": bw, "t_comm": t_comm}
                print(f"[{s} -> {r}] Latency: {lat:.4f} s | Bandwidth: {bw:.2f} B/s | T_comm: {t_comm:.4f} s")
    else:
        try:
            graph = network_graph(nodes)
        except Exception as e:
            print(f"error ({e}). Falling back to default link estimation.")
            graph = {
                a: {
                    b: {"latency": 0.003, "bandwidth": 1_250_000_000.0}
                    for b in nodes
                    if b != a
                }
                for a in nodes
            }

        for s in graph:
            for r in graph[s]:
                lat = graph[s][r].get("latency", 0.0)
                bw = graph[s][r].get("bandwidth", 1_250_000_000.0)
                t_comm = communication_time(
                    lat, bw, batch_size, seq_len, embed_dim
                )
                graph[s][r]["t_comm"] = t_comm
                print(f"[{s} -> {r}] Latency: {lat:.4f} s | Bandwidth: {bw:.2f} B/s | T_comm: {t_comm:.4f} s")

    node_a, node_b = nodes[0], nodes[1]
    lat_ab = graph[node_a][node_b]["latency"]
    bw_ab = graph[node_a][node_b]["bandwidth"]
    t_comm_ab = graph[node_a][node_b]["t_comm"]

# Dynamic Programming Scheduler 
print("\n--- Running Dynamic Programming Scheduler ---")
assignment, total_cost = dp_scheduler(
    numLayers=num_layers,
    numNodes=len(nodes),
    t_mlp=mlp_gpu,
    t_attn_gpu=attn_gpu,
    t_attn_cpu=attn_cpu,
    latency=lat_ab,
    bandwidth=bw_ab,
    batchSize=batch_size,
    seqLen=seq_len,
    embedDim=embed_dim,
    gpuMem=gpu_mem,
)

print(f"\nEnvironment Mode: {selected_env if selected_env else 'Custom IPs'}")
print(f"Inter-Node Latency: {lat_ab:.4f} seconds")
print(f"Communication Overhead (T_comm): {t_comm_ab:.4f} seconds")
print(f"Bandwidth: {bw_ab:.2f} bytes/second")
print(f"\nLayer Assignment per Node: {assignment}")
print(f"Total Cost (DP): {total_cost:.4f}s")