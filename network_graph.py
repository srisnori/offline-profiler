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


def network_graph(nodes, env=None):
    graph = {}
    for a in nodes:
        graph[a] = {}
        for b in nodes:
            if a == b:
                continue

            # Preset Environment Selected (E1 - E6)
            if env is not None:
                lat = 0.050 if env == "E6" else 0.003
                bw = get_bandwidth(sender=a, receiver=b, env=env)
            
            # Custom Real IPs -> Measure Live
            else:
                try:
                    lat = network_latency(a, b)
                except Exception:
                    lat = 0.003
                
                a_sub = ".".join(a.split(".")[:3])
                b_sub = ".".join(b.split(".")[:3])
                bw = 1_250_000_000.0 if a_sub == b_sub else 125_000_000.0

            graph[a][b] = {"latency": lat, "bandwidth": bw}

    return graph