import math
from performance_model import node_cost

def dp_scheduler(numLayers, numNodes, t_mlp, t_attn, latency, bandwidth, batchSize, seqLen, embedDim):
    INF = float("inf")

    dp = [[INF] * (numLayers + 1) for i in range(nodes + 1)]
    split = [[INF] * (numLayers + 1) for i in range(nodes + 1)]
    dp[0][0] = 0  # dp[nodes][layers] best way to put these many layers into these many nodes

    for i in range(1, nodes + 1): # which node we are on
        for l in range(numLayers + 1): # how many layers can u fit in that node
            for k in range(l + 1): # how many layers are in the current node
                prev = dp[i - 1][k] # look at previous answers and find the best solution for rest of layers
                if prev < INF:
                    continue
                
                layersNode = l - k
                cost = node_cost(numLayers, t_mlp, t_attn, latency, bandwidth, batch_size, seq_len, embed_dim)
                res = prev + cost

                if res < dp[i][l]:
                    dp[i][l] = res
                    split[i][l] = k
    
    layersAssigned = []
    l = layers

    for i in range(nodes, 0, -1):
        k = split[i][l]
        layersAssigned.append(l - k)
        l = k
    
    layersAssigned.reverse()
    return layersAssigned, dp[nodes][layers]