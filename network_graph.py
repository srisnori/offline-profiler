import json
from communication_time import communication_time
from itertools import combinations 
from network_latency import network_latency


def graph(nodes, chain=True):
    graph = {n: {} for n in nodes}
    if chain:
        pairs = list(zip(nodes[:-1], nodes[1:]))
    else:
        pairs = list(combinations(nodes, 2))

    for a, b in pairs:
        latency, bandwidth = measure_link(a, b)
        edge = {"latency": latency, "bandwidth": bandwidth}
        graph[a][b] = edge
        graph[b][a] = edge
    return graph


def path_cost(graph, path, batch_size, seq_len, embed_dim):
    total_sec = 0
    for a, b in zip(path[:-1], path[1:]):
        edge = graph[a][b]
        latency_sec = edge["latency"] / 1000
        total_sec += communication_time(latency_sec, edge["bandwidth"], batch_size, seq_len, embed_dim)
    return total_sec