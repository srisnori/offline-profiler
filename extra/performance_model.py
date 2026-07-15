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

def node_cost(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, latency, bandwidth, batch_size, seq_len, embed_dim, gpu_mem, micro_batches=1):
    alpha_i = calculate_offload_ratio(num_layers, batch_size, seq_len, embed_dim, gpu_mem)
    
    comp_total = compute_time(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, alpha_i)
    comp_per_batch = comp_total / micro_batches
    
    b = batch_size / micro_batches
    comm_per_batch = communication_time(latency, bandwidth, b, seq_len, embed_dim)
    
    return max(comp_per_batch, comm_per_batch)
