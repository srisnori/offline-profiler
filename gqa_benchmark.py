# fix this file

import torch, time

def GQA_CPU(embed_dim, batch_size, seq_len, num_heads):
    gqa = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    x = torch.rand(batch_size, seq_len, embed_dim)

    start = time.time()
    gqa(x, x, x)
    return time.time() - start

def GQA_GPU(embed_dim, batch_size, seq_len, num_heads):
    gqa = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).cuda()
    x = torch.rand(batch_size, seq_len, embed_dim).cuda()

    torch.cuda.synchronize()  # make sure all previous GPU jobs are done first
    start = time.time()
    gqa(x, x, x)
    torch.cuda.synchronize()
    end = time.time()
    return end - start