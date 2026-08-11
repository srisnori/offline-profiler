import ipaddress
import socket

import torch
from bandwidth import get_bandwidth
from communication_time import communication_time
from gqa_benchmark import GQA_CPU, GQA_GPU
from mha_benchmark import MHA_CPU, MHA_GPU
from mlp_benchmark import MLP_CPU, MLP_GPU
from network_graph import network_graph
from scheduler import dp_scheduler

# CPU versus GPU 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

# User Inputs 
model_name = input("Model: ").strip()
batch_size = int(input("Batch Size: "))
seq_len = int(input("Seq Length: "))
num_layers = int(input("Num Layers: "))
num_heads = int(input("Num Heads: "))
embed_dim = int(input("Embed Dim: "))
attention_mechanism = input("Attention (MHA/GQA/MLP): ").strip().lower()

# GPU allocation per node
if device == "cuda":
    default_vram = int(
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
    )
    gpu_mem_input = input(f"GPU Memory in GB (default {default_vram}): ").strip()
    gpu_mem = int(gpu_mem_input) if gpu_mem_input else default_vram
else:
    gpu_mem = 48

# IP Input & Validation 
raw_ips = input("Enter Distributed IPs (comma-separated): ").strip()
ip_list = [ip.strip() for ip in raw_ips.split(",") if ip.strip()]

valid_ips = []
for ip in ip_list:
    try:
        ipaddress.IPv4Address(ip)
        valid_ips.append(ip)
    except ValueError:
        print(f"[Error] '{ip}' is not a valid IPv4 address!")

if not valid_ips:
    raise ValueError("No valid IP addresses provided. Exiting.")

# Compute Benchmarks (CPU & GPU)
print(f"\n--- Running Compute Benchmarks ({device.upper()}) ---")

# MLP Benchmarks
mlp_cpu = MLP_CPU(embed_dim, batch_size, seq_len)
mlp_gpu = (
    MLP_GPU(embed_dim, batch_size, seq_len) if device == "cuda" else mlp_cpu
)

# Attention Benchmarks
if attention_mechanism == "mha":
    mech = "MHA"
    attn_cpu = MHA_CPU(embed_dim, batch_size, seq_len, num_heads)
    attn_gpu = (
        MHA_GPU(embed_dim, batch_size, seq_len, num_heads)
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
print(
    f"{mech} Time (CPU): {attn_cpu:.4f} s | {mech} Time (GPU): {attn_gpu:.4f} s"
)

layer_compute_gpu = attn_gpu + mlp_gpu
total_model_compute_gpu = layer_compute_gpu * num_layers
print(f"Single Layer GPU Time ({mech} + MLP): {layer_compute_gpu:.4f} s")
print(
    f"Total Model GPU Time ({num_layers} Layers): {total_model_compute_gpu:.4f} s"
)

# Network Pairwise Matrix Profiling 
print("\n--- Network Matrix Profiling ---")

if len(valid_ips) < 2:
    print("[Warning] Need at least 2 node IPs")
    lat_ab, bw_ab, t_comm_ab = 0.0, 1_250_000_000.0, 0.0
    graph = {ip: {} for ip in valid_ips}
else:
    try:
        graph = network_graph(valid_ips)
    except Exception as e:
        print(f"[Warning] Socket probe error ({e}). Falling back to static link estimation.")
        graph = {
            node_a: {
                node_b: {"latency": 0.003, "bandwidth": 1_250_000_000.0}
                for node_b in valid_ips
                if node_b != node_a
            }
            for node_a in valid_ips
        }

    # Calculate T_comm for ALL directed links in the network matrix
    for sender in graph:
        for receiver in graph[sender]:
            lat = graph[sender][receiver].get("latency", 0.0)
            bw = graph[sender][receiver].get("bandwidth", 1_250_000_000.0)

            t_comm = communication_time(
                lat, bw, batch_size, seq_len, embed_dim
            )
            graph[sender][receiver]["t_comm"] = t_comm

            print(
                f"[{sender} -> {receiver}] Latency: {lat:.4f} s | Bandwidth: {bw:.2f} B/s | T_comm: {t_comm:.4f} s"
            )

    # Primary link metrics passed into the DP scheduler summary
    node_a, node_b = valid_ips[0], valid_ips[1]
    lat_ab = graph[node_a][node_b]["latency"]
    bw_ab = graph[node_a][node_b]["bandwidth"]
    t_comm_ab = graph[node_a][node_b]["t_comm"]

# DP Scheduler  
print("\n--- Running Dynamic Programming Scheduler ---")
num_nodes = len(valid_ips)

assignment, total_cost = dp_scheduler(
    numLayers=num_layers,
    numNodes=num_nodes,
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

print(f"\nPipeline Bottleneck Link Latency: {lat_ab:.4f} seconds")
print(f"Communication Penalty (T_comm): {t_comm_ab:.4f} seconds")
print(f"Interconnect Bandwidth: {bw_ab:.2f} bytes/second")
print(f"\nLayer Assignment per Node: {assignment}")
print(f"Total Cost (DP): {total_cost:.4f}s")