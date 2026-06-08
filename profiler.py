import threading, time

from network_latency import network_latency
from communication_time import communication_time
from mha_benchmark import MHA_CPU, MHA_GPU
from mlp_benchmark import MLP_CPU, MLP_GPU
from network_bandwidth.sender import send_bandwidth
from network_bandwidth.receiver import receive_bandwidth
from gqa_benchmark import GQA_CPU, GQA_GPU

model_name = "gpt-2"
batch_size = 8
seq_len = 512
gpu = {"type": "A100", "memory_gb": 40}
distributed_sites = ["10.0.0.1", "10.0.0.2"]
data_size_bytes = 50 * 1024 * 1024
payload = b"x" * data_size_bytes

# -------------------
# COMPUTE
# -------------------
mlp_cpu = MLP_CPU(embed_dim, batch_size, seq_len)
mlp_gpu = MLP_GPU(embed_dim, batch_size, seq_len)
mha_cpu = MHA_CPU(embed_dim, batch_size, seq_len, num_heads)
mha_gpu = MHA_GPU(embed_dim, batch_size, seq_len, num_heads)
gqa_cpu = GQA_CPU(embed_dim, batch_size, seq_len, num_heads)
gqa_gpu = GQA_GPU(embed_dim, batch_size, seq_len, num_heads)
network = network_latency(ip)

# -------------------
# START RECEIVER FIRST
# -------------------
receiver_thread = threading.Thread(
    target=receive_bandwidth,
    args=(port,),
    daemon=True
)
receiver_thread.start()
time.sleep(0.5) 

# -------------------
# SEND DATA
# -------------------
bandwidth = send_bandwidth(ip, data, port)
receiver_thread.join()

# -------------------
# COMMUNICATION MODEL
# -------------------
communication = communication_time(network, bandwidth, data_size)

# -------------------
# OUTPUT
# -------------------
print(f"MLP Time (CPU): {mlp_cpu:.4f} seconds")
print(f"MHA Time (CPU): {mha_cpu:.4f} seconds")
print(f"GQA Time (CPU): {gqa_cpu:.4f} seconds")
print(f"Network Latency: {network:.4f} seconds")
print(f"Communication Time: {communication:.4f} seconds")
print(f"Bandwidth: {bandwidth:.2f} bytes/second")