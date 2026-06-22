from communication_time import communication_time

def compute_time(num_layers, t_mlp, t_attn):
    return num_layers * (t_mlp + t_attn)

def node_cost(num_layers, t_mlp, t_attn, latency, bandwidth, batch_size, seq_len, embed_dim):
    comp = compute_time(num_layers, t_mlp, t_attn)
    comm = communication_time(latency, bandwidth, batch_size, seq_len, embed_dim)
    return comp + comm