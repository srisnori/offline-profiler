import torch, time

def MHA_CPU(embed_dim, batch_size, seq_len, num_heads):
    mha = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    x = torch.rand(batch_size, seq_len, embed_dim)

    start = time.time()
    mha(x, x, x)
    return time.time() - start

def MHA_GPU(embed_dim, batch_size, seq_len, num_heads):
    mha = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).cuda().half()
    x = torch.rand(batch_size, seq_len, embed_dim).cuda().half()

    for _ in range(3):
        mha(x, x, x)
    torch.cuda.synchronize()

    runs = 10                                  
    start = time.time()
    for _ in range(runs):
        mha(x, x, x)
    torch.cuda.synchronize()
    return (time.time() - start) / runs

