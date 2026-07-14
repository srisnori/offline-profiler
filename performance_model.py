from communication_time import communication_time

def calculate_offload_ratio(num_layers, batch_size, seq_len, embed_dim, gpu_mem):
    bytes_per_element = 2
    activation_size = (batch_size * seq_len * embed_dim * bytes_per_element)
    total_activation_memory = (activation_size * num_layers) / (1024 ** 3)
    available_gpu_memory = gpu_mem * 0.8  
    if total_activation_memory <= available_gpu_memory:
        return 0.0
    overflow = (total_activation_memory - available_gpu_memory) / total_activation_memory
    return min(max(overflow, 0.0), 1.0)

def compute_time(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, alpha_i):
    t_block = (t_mlp + (1 - alpha_i) * t_attn_gpu + alpha_i * t_attn_cpu)
    return num_layers * t_block

def node_cost(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, latency, bandwidth, batch_size, seq_len, embed_dim, gpu_mem):
    alpha_i = calculate_offload_ratio(num_layers, batch_size, seq_len, embed_dim, gpu_mem)
    comp = compute_time(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, alpha_i)
    comm = communication_time(latency, bandwidth, batch_size, seq_len, embed_dim)
    return comp + comm