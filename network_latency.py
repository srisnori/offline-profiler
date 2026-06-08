import time, socket, time

def network_latency(ip):
    start = time.time()

    try: 
        socket.create_connection((ip, 443), timeout=2)
    except:
        pass
        
    end = time.time()
    return end - start