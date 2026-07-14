GBPS = 1_000_000_000
MBPS = 1_000_000

def mbps_to_bytes(mbps):
    return (mbps * MBPS) / 8

ENVIRONMENTS = {
    "E1": {"description": "45 Gbps single-cluster", "bandwidth": mbps_to_bytes(45000),},
    "E2": {"description": "500 Mbps homogeneous", "bandwidth": mbps_to_bytes(500),},
    "E3": {"description": "250 Mbps homogeneous", "bandwidth": mbps_to_bytes(250),},
    "E4": {"description": "125 Mbps homogeneous", "bandwidth": mbps_to_bytes(125),},
    "E5": {"description": "20 Mbps homogeneous", "bandwidth": mbps_to_bytes(20),},
}

E6 = {
    ("California", "New Jersey"): 312,
    ("California", "Canada"): 280,
    ("New Jersey", "California"): 347,
    ("New Jersey", "Canada"): 643,
    ("Canada", "California"): 305,
    ("Canada", "New Jersey"): 577,
}


def get_bandwidth(env, sender=None, receiver=None):
    env = env.upper()
    if env != "E6":
        return ENVIRONMENTS[env]["bandwidth"]

    if sender is None or receiver is None:
        raise ValueError("E6 requires sender and receiver regions.")
    mbps = E6[(sender, receiver)]
    return mbps_to_bytes(mbps)