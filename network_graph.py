import threading
import time

from network_latency import network_latency
from network_bandwidth.sender import send_bandwidth
from network_bandwidth.receiver import receive_bandwidth

def measure_link(ip, port, data_size):
    data = b"x" * data_size
    latency = network_latency(ip)
    receiver_thread = threading.Thread(target=receive_bandwidth, args=(port,), daemon=True)
    receiver_thread.start()
    time.sleep(0.5)

    bandwidth = send_bandwidth(ip, data, port)
    receiver_thread.join()
    return {"latency": latency, "bandwidth": bandwidth}


def network_graph(nodes, latency, bandwidth):
    graph = {}
    for node in nodes:
        graph[node] = {}

    for a, b in zip(nodes[:-1], nodes[1:]):
        metrics = measure_link(b)
        graph[a][b] = metrics
    return graph