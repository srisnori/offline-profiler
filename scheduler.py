import math
from performance_model import node_cost

def dp_scheduler(numLayers, numNodes, t_mlp, t_attn, latency, bandwidth, batchSize, seqLen, embedDim, gpuMem):
    INF = float("inf")

    dp = [[INF] * (numLayers + 1) for i in range(numNodes + 1)]
    split = [[-1] * (numLayers + 1) for i in range(numNodes + 1)]
    dp[0][0] = 0  # dp[nodes][layers] best way to put these many layers into these many nodes

    for i in range(1, numNodes + 1): # which node we are on
        for l in range(numLayers + 1): # how many layers can u fit in that node
            for k in range(l + 1): # how many layers are in the current node
                prev = dp[i - 1][k] # look at previous answers and find the best solution for rest of layers
                if prev == INF:
                    continue
                
                layersNode = l - k
                if layersNode == 0:
                    continue
                cost = node_cost(layersNode, t_mlp, t_attn, latency, bandwidth, batchSize, seqLen, embedDim, gpuMem)
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