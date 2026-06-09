# measures only one TCP transfer connection
import socket

def receive_bandwidth(port):
    s = socket.socket()
    s.bind(('0.0.0.0', port))
    s.listen(1)

    conn, addr = s.accept()
    while conn.recv(1024):
        pass

    conn.close()
    s.close()