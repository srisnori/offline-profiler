from network_latency import network_latency
from communication_time import communication_time
from mha_benchmark import MHA_CPU, MHA_GPU
from mlp_benchmark import MLP_CPU, MLP_GPU
from gqa_benchmark import GQA_CPU, GQA_GPU
from bandwidth import get_bandwidth

model_name = input("Model: ")
batch_size = int(input("Batch Size: "))
seq_len = int(input("Seq Length: "))
num_heads = int(input("Num Heads: "))
embed_dim = int(input("Embed Dim: "))
attention_mechanism = input("Attention (MHA/GQA/MLP): ").strip().lower()
gpu_type = input("GPU: ")
gpu_mem = int(input("GPU Memory (GB): "))
env = input("Environment (E1-E6): ").strip().upper()

if env == "E6":
    sender = input("Sender Region (California/New Jersey/Canada): ").strip()
    receiver = input("Receiver Region (California/New Jersey/Canada): ").strip()
    bandwidth = get_bandwidth(env, sender, receiver)
else:
    bandwidth = get_bandwidth(env)

gpu = {"type": gpu_type, "memory_gb": gpu_mem}

mlp_cpu = MLP_CPU(embed_dim, batch_size, seq_len)
mlp_gpu = MLP_GPU(embed_dim, batch_size, seq_len)

if attention_mechanism == "mha":
    mech = "MHA"
    attn_cpu = MHA_CPU(embed_dim, batch_size, seq_len, num_heads)
    attn_gpu = MHA_GPU(embed_dim, batch_size, seq_len, num_heads)

elif attention_mechanism == "gqa":
    mech = "GQA"
    attn_cpu = GQA_CPU(embed_dim, batch_size, seq_len, num_heads)
    attn_gpu = GQA_GPU(embed_dim, batch_size, seq_len, num_heads)

elif attention_mechanism == "mlp":
    mech = "MLP"
    attn_cpu = mlp_cpu
    attn_gpu = mlp_gpu

else:
    raise ValueError("Attention must be mha, gqa, or mlp.")

network = network_latency(env)
communication = communication_time(network, bandwidth, batch_size, seq_len, embed_dim)

print("")
print(f"MLP Time (CPU): {mlp_cpu:.4f} s")
print(f"MLP Time (GPU): {mlp_gpu:.4f} s")
print(f"{mech} Time (CPU): {attn_cpu:.4f} s")
print(f"{mech} Time (GPU): {attn_gpu:.4f} s")
print(f"Network Latency: {network:.4f} s")
print(f"Bandwidth: {bandwidth:.2f} bytes/sec")
print(f"Communication Time (T_comm): {communication:.4f} s")
