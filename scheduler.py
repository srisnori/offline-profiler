import math
from performance_model import node_cost

def dp_scheduler(numLayers, numNodes, t_mlp, t_attn_gpu, t_attn_cpu, latency, bandwidth, batchSize, seqLen, embedDim, gpuMem, minGpuMem=4.0):
    INF = float("inf")


    if isinstance(gpuMem, (int, float)):
        nodes_gpu_config = [[float(gpuMem)] for _ in range(numNodes)]
    elif isinstance(gpuMem, list):
        if len(gpuMem) != numNodes:
            raise ValueError(f"Length of gpuMem ({len(gpuMem)}) must match numNodes ({numNodes}).")
        nodes_gpu_config = []
        for item in gpuMem:
            if isinstance(item, (int, float)):
                nodes_gpu_config.append([float(item)])
            elif isinstance(item, list):
                nodes_gpu_config.append([float(g) for g in item])
            else:
                raise TypeError("Each node GPU configuration must be a number or list of numbers.")
    else:
        raise TypeError("gpuMem must be a float, int, or list.")

    # check per-GPU minimum memory 
    for node_idx, gpus in enumerate(nodes_gpu_config):
        for gpu_idx, mem in enumerate(gpus):
            if mem < minGpuMem:
                raise ValueError(
                    f"Node {node_idx} GPU {gpu_idx} has {mem:.2f} GB VRAM, "
                    f"which is below the minimum required threshold of {minGpuMem:.2f} GB."
                )

    # calculate total VRAM per node
    node_vram_totals = [sum(gpus) for gpus in nodes_gpu_config]

    dp = [[INF] * (numLayers + 1) for i in range(numNodes + 1)]
    split = [[-1] * (numLayers + 1) for i in range(numNodes + 1)]
    dp[0][0] = 0  # dp[nodes][layers] best way to put these many layers into these many nodes

    for i in range(1, numNodes + 1): # which node we are on
        for l in range(1, numLayers + 1): # how many layers can u fit in that node
            for k in range(l + 1): # how many layers are in the current node
                prev = dp[i - 1][k] # look at previous answers and find the best solution for rest of layers
                if prev == INF:
                    continue
                
                layersNode = l - k
                if layersNode == 0:
                    continue
                cost = node_cost(layersNode, t_mlp, t_attn_gpu, t_attn_cpu, latency, bandwidth, batchSize, seqLen, embedDim, current_node_mem)
                
                if cost == INF: continue
                res = max(prev, cost)
                if res < dp[i][l]:
                    dp[i][l] = res
                    split[i][l] = k

    if dp[numNodes][numLayers] == INF:
        print("No layer assignment found.")
        return [], INF

    layersAssigned = []
    l = numLayers

    for i in range(numNodes, 0, -1):
        k = split[i][l]
        layersAssigned.append(l - k)
        l = k
    
    layersAssigned.reverse()
    return layersAssigned, dp[numNodes][numLayers]