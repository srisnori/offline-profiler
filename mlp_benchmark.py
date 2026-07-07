import torch, time

def MLP_CPU(embed_dim, batch_size, seq_len):
    mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim))
    x = torch.rand(batch_size, seq_len, embed_dim)

    start = time.time()
    mlp(x)
    return time.time() - start

def MLP_GPU(embed_dim, batch_size, seq_len):
    mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim)).cuda().half()
    x = torch.rand(batch_size, seq_len, embed_dim).cuda().half()

    for _ in range(3):
        mlp(x)
    torch.cuda.synchronize()

    start = time.time()
    mlp(x)
    torch.cuda.synchronize()
    return time.time() - start
