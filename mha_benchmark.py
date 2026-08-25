import time
import torch

def MHA_CPU(embed_dim, batch_size, seq_len, num_heads):
  mha = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
  x = torch.rand(batch_size, seq_len, embed_dim)

  with torch.no_grad():
    start = time.perf_counter()
    mha(x, x, x)
    return time.perf_counter() - start


def MHA_GPU(embed_dim, batch_size, seq_len, num_heads):
  mha = (torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).cuda().half())
  x = torch.rand(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)

  with torch.no_grad():
    for _ in range(3):
      mha(x, x, x)
    torch.cuda.synchronize()

    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
      mha(x, x, x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs