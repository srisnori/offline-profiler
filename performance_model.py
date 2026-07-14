from communication_time import communication_time

def compute_time(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, alpha_i):
    t_block = t_mlp + (1 - alpha_i) * t_attn_gpu + alpha_i * t_attn_cpu
    return num_layers * t_block

def node_cost(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, latency, bandwidth, batch_size, seq_len, embed_dim, gpu_mem):
    alpha_i = calculate_offload_ratio(num_layers, batch_size, seq_len, embed_dim, gpu_mem) 
    comp = compute_time(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, alpha_i)
    comm = communication_time(latency, bandwidth, batch_size, seq_len, embed_dim) 
    return max(comp, comm) 
