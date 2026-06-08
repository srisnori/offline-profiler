import torch, time

def MLP_CPU(embed_dim, batch_size, seq_len):
    mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim))
    x = torch.rand(batch_size, seq_len, embed_dim)

    start = time.time()
    mlp(x)
    return time.time() - start

def MLP_GPU(embed_dim, batch_size, seq_len):
    mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim)).cuda()
    x = torch.rand(batch_size, seq_len, embed_dim).cuda()

    torch.cuda.synchronize()  # make sure all previous GPU jobs are done first
    start = time.time()
    mlp(x)
    torch.cuda.synchronize()
    end = time.time()
    return end - start