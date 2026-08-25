from communication_time import communication_time


def calculate_offload_ratio(num_layers, batch_size, seq_len, embed_dim, gpu_mem):
  bytes_per_element = 2  # FP16
  params_per_layer = 12 * (embed_dim**2)
  weight_mem_gb = (params_per_layer * bytes_per_element * num_layers) / (1024**3)

  # Activation & KV-cache memory
  activation_size = batch_size * seq_len * embed_dim * bytes_per_element
  act_mem_gb = (activation_size * num_layers) / (1024**3)

  total_required_mem = weight_mem_gb + act_mem_gb
  available_gpu_memory = gpu_mem * 0.85  # 85% usable VRAM budget

  if total_required_mem <= available_gpu_memory:
    return 0.0

  # Offload ratio: fraction of compute/KV pushed to CPU host RAM
  overflow = (total_required_mem - available_gpu_memory) / total_required_mem
  return min(max(overflow, 0.0), 1.0)


def compute_time(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, alpha_i): # GPU and CPU attention execution based on offload ratio alpha_i
  t_block = t_mlp + (1.0 - alpha_i) * t_attn_gpu + alpha_i * t_attn_cpu
  return num_layers * t_block


def node_cost(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, latency, bandwidth, batch_size, seq_len, embed_dim, gpu_mem, micro_batches=1,):
  alpha_i = calculate_offload_ratio(num_layers, batch_size, seq_len, embed_dim, gpu_mem)
  comp_total = compute_time(num_layers, t_mlp, t_attn_gpu, t_attn_cpu, alpha_i)
  comp_per_batch = comp_total / micro_batches

  b = batch_size / micro_batches
  comm_per_batch = communication_time(latency, bandwidth, b, seq_len, embed_dim)
  return max(comp_per_batch, comm_per_batch)