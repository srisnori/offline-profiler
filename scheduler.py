import math
from performance_model import node_cost

def dp_scheduler(numLayers, numNodes, t_mlp, t_attn_gpu, t_attn_cpu, latency, bandwidth, batchSize, seqLen, 
    embedDim, num_gpus, gpu_vram, minGpuMem=4.0):
    INF = float("inf")

    # check uniform GPU VRAM size
    gpu_vram = float(gpu_vram)
    if gpu_vram < minGpuMem:
        raise ValueError(
            f"Configured GPU VRAM ({gpu_vram:.2f} GB) is below the minimum "
            f"required threshold of {minGpuMem:.2f} GB."
        )

    if isinstance(num_gpus, int):
        node_gpu_counts = [num_gpus] * numNodes
    elif isinstance(num_gpus, list):
        if len(num_gpus) != numNodes:
            raise ValueError(f"Length of num_gpus ({len(num_gpus)}) must match numNodes ({numNodes}).")
        node_gpu_counts = [int(count) for count in num_gpus]
    else:
        raise TypeError("num_gpus must be an int or a list of ints.")

    for node_idx, count in enumerate(node_gpu_counts):
        if count < 1:
            raise ValueError(f"Node {node_idx} must have at least 1 GPU (got {count}).")

    # total VRAM per node
    node_vram_totals = [count * gpu_vram for count in node_gpu_counts]

    dp = [[INF] * (numLayers + 1) for _ in range(numNodes + 1)]
    split = [[-1] * (numLayers + 1) for _ in range(numNodes + 1)]
    dp[0][0] = 0.0

    # DP Recurrence
    for i in range(1, numNodes + 1):
        current_node_mem = node_vram_totals[i - 1]

        if isinstance(latency, list) and isinstance(bandwidth, list):
            cur_lat = latency[i - 1] if i - 1 < len(latency) else 0.0
            cur_bw = bandwidth[i - 1] if i - 1 < len(bandwidth) else INF
        else:
            cur_lat = latency
            cur_bw = bandwidth

        for l in range(1, numLayers + 1):
            for k in range(l + 1):
                prev = dp[i - 1][k]
                if prev == INF:
                    continue
                
                layersNode = l - k
                if layersNode == 0:
                    continue

                cost = node_cost(layersNode, t_mlp, t_attn_gpu, t_attn_cpu, cur_lat, cur_bw, batchSize, seqLen, embedDim, current_node_mem)
                
                if cost == INF: 
                    continue

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