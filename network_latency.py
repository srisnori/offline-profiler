import os, platform, socket, time

def network_latency(sender, receiver, port=22, timeout=2.0):
    try:
        start = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((receiver, port))
        sock.close()
        end = time.perf_counter()
        return end - start
    except (socket.timeout, ConnectionRefusedError, OSError):
        pass

    param = "-n 1" if platform.system().lower() == "windows" else "-c 1"
    start = time.perf_counter()
    response = os.system(
        f"ping {param} -w 2000 {receiver} > /dev/null 2>&1"
        if platform.system().lower() != "windows"
        else f"ping {param} {receiver} > NUL"
    )
    end = time.perf_counter()

    if response == 0:
        return end - start

    print(f"[Warning] Host {receiver} is unreachable.")
    return float("inf")