def communication_time(latency, bandwidth, batch_size, seq_len, embed_dim):
    activation_size = seq_len * embed_dim * 4
    comm = latency + ((batch_size * activation_size) / bandwidth)
    return comm