import time
import torch

def MLP_CPU(embed_dim, batch_size, seq_len):
  mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim),)
  x = torch.rand(batch_size, seq_len, embed_dim)

  with torch.no_grad():
    start = time.perf_counter()
    mlp(x)
    return time.perf_counter() - start


def MLP_GPU(embed_dim, batch_size, seq_len):
  mlp = (torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim),).cuda().half())
  x = torch.rand(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)

  with torch.no_grad():
    for _ in range(3):
      mlp(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    mlp(x)
    torch.cuda.synchronize()
    return time.perf_counter() - start