GBPS = 1_000_000_000
MBPS = 1_000_000


def mbps_to_bytes(mbps):
    return (mbps * MBPS) / 8


ENVIRONMENTS = {
    "E1": {"description": "45 Gbps single-cluster", "bandwidth": mbps_to_bytes(45000)},
    "E2": {"description": "500 Mbps homogeneous", "bandwidth": mbps_to_bytes(500)},
    "E3": {"description": "250 Mbps homogeneous", "bandwidth": mbps_to_bytes(250)},
    "E4": {"description": "125 Mbps homogeneous", "bandwidth": mbps_to_bytes(125)},
    "E5": {"description": "20 Mbps homogeneous", "bandwidth": mbps_to_bytes(20)},
}

E6 = {
    ("California", "New Jersey"): 312,
    ("California", "Canada"): 280,
    ("New Jersey", "California"): 347,
    ("New Jersey", "Canada"): 643,
    ("Canada", "California"): 305,
    ("Canada", "New Jersey"): 577,
}

# Chameleon Cloud bare-metal interconnects default to 10 Gbps (10,000 Mbps = 1.25 GB/s)
DEFAULT_CHAMELEON_MBPS = 10000


def get_bandwidth(sender=None, receiver=None, env=None, default_mbps=DEFAULT_CHAMELEON_MBPS):
    """
    Returns bandwidth in bytes/sec.
    Supports preset environments (E1-E6) or explicit sender/receiver pairs.
    """
    if env:
        env = env.upper()
        if env in ENVIRONMENTS:
            return ENVIRONMENTS[env]["bandwidth"]
        elif env == "E6":
            if sender is None or receiver is None:
                raise ValueError("E6 requires sender and receiver region names.")
            mbps = E6.get((sender, receiver))
            if mbps is None:
                raise ValueError(f"Unknown E6 region pair: ({sender}, {receiver})")
            return mbps_to_bytes(mbps)
        else:
            raise ValueError(f"Unknown environment preset: {env}")

    if sender and receiver:
        # Check if region names were passed directly
        if (sender, receiver) in E6:
            return mbps_to_bytes(E6[(sender, receiver)])

        # Return default Chameleon link speed for custom IP pairs
        return mbps_to_bytes(default_mbps)

    raise ValueError("Must specify either a preset environment 'env' or valid IP pairs.")