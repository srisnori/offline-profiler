import subprocess
import re
import socket
import time

def network_latency(sender, receiver):
    if sender == receiver:
        return 0.0

    target = str(receiver).strip()

    try:
        cmd = ["ping", "-c", "1", "-W", "1", target]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1.2  # Python process kill deadline
        )
        if proc.returncode == 0:
            match = re.search(r"rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/", proc.stdout)
            if match:
                return float(match.group(1)) / 1000.0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception):
        pass

    try:
        t0 = time.time()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)  # Hard ceiling on connect
        s.connect((target, 22))
        lat = time.time() - t0
        s.close()
        return lat
    except Exception:
        pass

    print(f"[Warning] Host {target} unreachable. Applied 0.080s default penalty.")
    return 0.080