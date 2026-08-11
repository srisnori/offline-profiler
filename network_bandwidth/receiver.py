# measures only one TCP transfer connection
import socket

def receive_bandwidth(port=5001):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(1)

    conn, addr = s.accept()
    total_bytes = 0
    while True:
        data = conn.recv(1024 * 1024)
        if not data:
            break
        total_bytes += len(data)

    conn.close()
    s.close()