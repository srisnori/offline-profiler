import threading, time

from network_bandwidth.receiver import receive_bandwidth
from network_bandwidth.sender import send_bandwidth
from network_latency import network_latency
from bandwidth import get_bandwidth

def measure_link(sender, receiver, port=5001, data_size=10_000_000):
    data = b"x" * data_size
    latency = network_latency(sender, receiver)
    receiver_thread = threading.Thread(target=receive_bandwidth, args=(port,), daemon=True)
    receiver_thread.start()
    time.sleep(0.5)

    bandwidth = send_bandwidth(receiver, data, port)
    receiver_thread.join()
    return {"latency": latency, "bandwidth": bandwidth}


def network_graph(nodes):
    graph = {}
    for a in nodes:
        graph[a] = {}
        for b in nodes:
            if a == b:
                continue
            graph[a][b] = {"latency": network_latency(a, b), "bandwidth": get_bandwidth(a, b)}
    return graph