# preset envs for testing
GBPS = 1_000_000_000
MBPS = 1_000_000

def mbps_to_bytes(mbps):
    return (mbps * MBPS) / 8  # Mbps to Bytes per second

E6_RATES = {
    ("California", "New Jersey"): 312,
    ("California", "Canada"): 280,
    ("New Jersey", "California"): 347,
    ("New Jersey", "Canada"): 643,
    ("Canada", "California"): 305,
    ("Canada", "New Jersey"): 577,
}

ENVIRONMENTS = {
    "E1": {
        "description": "45 Gbps single-cluster",
        "bandwidth": mbps_to_bytes(45000),
    },
    "E2": {
        "description": "500 Mbps homogeneous",
        "bandwidth": mbps_to_bytes(500),
    },
    "E3": {
        "description": "250 Mbps homogeneous",
        "bandwidth": mbps_to_bytes(250),
    },
    "E4": {
        "description": "125 Mbps homogeneous",
        "bandwidth": mbps_to_bytes(125),
    },
    "E5": {
        "description": "20 Mbps homogeneous",
        "bandwidth": mbps_to_bytes(20),
    },
    "E6": {
        "description": "Heterogeneous WAN matrix",
        "matrix": {
            pair: mbps_to_bytes(mbps) for pair, mbps in E6_RATES.items()
        },
    },
}

DEFAULT_LAN_MBPS = 10_000  # 10 Gbps LAN default

# use envs or user inputs
def get_bandwidth(sender=None, receiver=None, env=None):
    # Preset environment selected
    if env:
        env = env.upper()
        if env == "E6":
            if sender is None or receiver is None:
                raise ValueError("E6 requires sender and receiver region names.")
            
            # E6
            mbps = E6_RATES.get((sender, receiver))
            if mbps is None:
                raise ValueError(f"Unknown E6 region pair: ({sender}, {receiver})")
            return mbps_to_bytes(mbps)
            
        elif env in ENVIRONMENTS:
            return ENVIRONMENTS[env]["bandwidth"]
        else:
            raise ValueError(f"Unknown environment preset: {env}")

    # Live IP pairs 
    if sender and receiver:
        if (sender, receiver) in E6_RATES:
            return mbps_to_bytes(E6_RATES[(sender, receiver)])
        return mbps_to_bytes(DEFAULT_LAN_MBPS)

    raise ValueError("Must specify either a preset environment 'env' or valid sender/receiver pairs.")