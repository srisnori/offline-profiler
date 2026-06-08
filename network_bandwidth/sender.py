import time, socket

def send_bandwidth(ip, data, port):
    s = socket.socket()
    s.connect((ip, port))

    start = time.time()
    s.sendall(data)
    s.shutdown(socket.SHUT_WR) 
    s.close()
    
    return (len(data) / (time.time() - start))