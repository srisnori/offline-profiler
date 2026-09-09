import math
from bandwidth import get_bandwidth
from network_latency import network_latency

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
                    if lat is None or math.isinf(lat) or math.isnan(lat) or lat <= 0:
                        lat = 0.080
                except Exception:
                    lat = 0.080
                
                try:
                    a_sub = ".".join(a.split(".")[:3])
                    b_sub = ".".join(b.split(".")[:3])
                    bw = 1_250_000_000.0 if a_sub == b_sub else 125_000_000.0
                except Exception:
                    bw = 125_000_000.0

            graph[a][b] = {"latency": lat, "bandwidth": bw}

    return graph