import threading, time

from network_latency import network_latency
from communication_time import communication_time
from mha_benchmark import MHA_CPU, MHA_GPU
from mlp_benchmark import MLP_CPU, MLP_GPU
from network_bandwidth.sender import send_bandwidth
from network_bandwidth.receiver import receive_bandwidth
from gqa_benchmark import GQA_CPU, GQA_GPU
from performance_model import node_cost
from scheduler import dp_scheduler

# configs
model_name          = input("Model: ")
batch_size          = int(input("Batch Size: "))
seq_len             = int(input("Seq Length: "))
num_layers          = int(input("Num Layers: "))
num_heads           = int(input("Num Heads: "))
embed_dim           = int(input("Embed Dim: "))
attention_mechanism = input("Attention (MHA/GQA/MLP): ")
gpu_type            = input("GPU: ")
gpu_mem             = int(input("GPU Memory (GB): "))
ips                 = input("Distributed IPs (space separated): ").split()

# setup
gpu       = {"type": gpu_type, "memory_gb": gpu_mem}
data_size = 50 * 1024 * 1024
data      = b"x" * data_size
ip        = ips[0]
port      = 5001

# benchmark MLP
mlp_cpu = MLP_CPU(embed_dim, batch_size, seq_len)
mlp_gpu = MLP_GPU(embed_dim, batch_size, seq_len)

# benchmark attention mechanism
mech = attention_mechanism.upper()
if attention_mechanism.lower() == "mha":
    attn_cpu = MHA_CPU(embed_dim, batch_size, seq_len, num_heads)
    attn_gpu = MHA_GPU(embed_dim, batch_size, seq_len, num_heads)
elif attention_mechanism.lower() == "gqa":
    attn_cpu = GQA_CPU(embed_dim, batch_size, seq_len, num_heads)
    attn_gpu = GQA_GPU(embed_dim, batch_size, seq_len, num_heads)
elif attention_mechanism.lower() == "mlp":
    attn_cpu = mlp_cpu    
    attn_gpu = mlp_gpu
else:
    raise ValueError(f"Unknown: {attention_mechanism}. Choose mha, gqa, or mlp.")

# network
network = network_latency(ip)

receiver_thread = threading.Thread(target=receive_bandwidth, args=(port,), daemon=True)
receiver_thread.start()
time.sleep(0.5)
bandwidth = send_bandwidth(ip, data, port)
receiver_thread.join()

communication = communication_time(network, bandwidth, batch_size, seq_len, embed_dim)
layers_assignment, total_cost = dp_scheduler(numLayers = num_layers, numNodes = len(ips), t_mlp = mlp_gpu, t_attn = attn_gpu, latency = network, bandwidth = bandwidth, batchSize = batch_size, seqLen = seq_len, embedDim = embed_dim)

# outputs
print("")
print("")
print(f"MLP Time  (CPU): {mlp_cpu:.4f} seconds")
print(f"MLP Time  (GPU): {mlp_gpu:.4f} seconds")
print(f"{mech} Time  (CPU): {attn_cpu:.4f} seconds")  
print(f"{mech} Time  (GPU): {attn_gpu:.4f} seconds")  
print(f"Network Latency: {network:.4f} seconds")
print(f"Communication Time: {communication:.4f} seconds")
print(f"Bandwidth: {bandwidth:.2f} bytes/second")
print(f"Layer Assignment per Node: {layers_assignment}")
print(f"Total Cost (DP): {total_cost:.4f}s")