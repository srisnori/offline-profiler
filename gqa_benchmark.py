import torch
import time
import torch.nn.functional as F

def GQA_CPU(embed_dim, batch_size, seq_len, num_heads, num_kv_heads):
    head_dim = embed_dim // num_heads
    num_groups = num_heads // num_kv_heads

    q = torch.rand(batch_size, num_heads, seq_len, head_dim)
    k = torch.rand(batch_size, num_kv_heads, seq_len, head_dim)
    v = torch.rand(batch_size, num_kv_heads, seq_len, head_dim)

    k_expanded = k.repeat_interleave(num_groups, dim=1)
    v_expanded = v.repeat_interleave(num_groups, dim=1)

    start = time.perf_counter()
    F.scaled_dot_product_attention(q, k_expanded, v_expanded)
    return time.perf_counter() - start


def GQA_GPU(embed_dim, batch_size, seq_len, num_heads, num_kv_heads):
    head_dim = embed_dim // num_heads
    num_groups = num_heads // num_kv_heads

    q = torch.rand(batch_size, num_heads, seq_len, head_dim, device="cuda").half()
    k = torch.rand(batch_size, num_kv_heads, seq_len, head_dim, device="cuda").half()
    v = torch.rand(batch_size, num_kv_heads, seq_len, head_dim, device="cuda").half()

    k_expanded = k.repeat_interleave(num_groups, dim=1)
    v_expanded = v.repeat_interleave(num_groups, dim=1)

    for _ in range(3):
        F.scaled_dot_product_attention(q, k_expanded, v_expanded)
    torch.cuda.synchronize()

    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        F.scaled_dot_product_attention(q, k_expanded, v_expanded)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs