import threading, time
from transformers import AutoConfig

from network_latency import network_latency
from communication_time import communication_time
from mha_benchmark import MHA_CPU, MHA_GPU
from mlp_benchmark import MLP_CPU, MLP_GPU
from network_bandwidth.sender import send_bandwidth
from network_bandwidth.receiver import receive_bandwidth
from gqa_benchmark import GQA_CPU, GQA_GPU

# configs
model_name = input("Model: ")
batch_size = int(input("Batch Size: "))
seq_len = int(input("Seq length: "))
attention_mechanism = input("Attention (MHA/GQA/MQA): ")
gpu_type = input("GPU: ")
gpu_mem = int(input("GPU Memory (GB): "))
ips = input("Distributed IPs (space-separated): ").split()

# system setup
gpu = {"type": gpu_type, "memory_gb": gpu_mem}
data_size = 50 * 1024 * 1024
data = b"x" * data_size
ip = ips[0]
port = 5001

# derive from inputs
cfg = AutoConfig.from_pretrained(model_name)
embed_dim = cfg.n_embd
num_heads = cfg.n_head
num_layers = cfg.n_layer


# compute
mlp_cpu = MLP_CPU(embed_dim, batch_size, seq_len)
mha_cpu = MHA_CPU(embed_dim, batch_size, seq_len, num_heads)
network = network_latency(ip)

receiver_thread = threading.Thread(target=receive_bandwidth, args=(port,), daemon=True)
receiver_thread.start()
time.sleep(0.5) 
bandwidth = send_bandwidth(ip, data, port)
receiver_thread.join()

communication = communication_time(network, bandwidth, data_size)

# outpts
print(f"MLP Time (CPU): {mlp_cpu:.4f} seconds")
print(f"MHA Time (CPU): {mha_cpu:.4f} seconds")
print(f"Network Latency: {network:.4f} seconds")
print(f"Communication Time: {communication:.4f} seconds")
print(f"Bandwidth: {bandwidth:.2f} bytes/second")