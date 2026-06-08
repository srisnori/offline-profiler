def communication_time(latency, bandwidth, data_size):
    latency = latency 
    bandwidth = bandwidth
    dataSize = data_size  

    comm = latency + (dataSize / bandwidth)

    return comm