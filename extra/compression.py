import time
import zstandard as zstd

def byte_split(data: bytes) -> bytes:
    if len(data) % 2 != 0:
        raise ValueError("FP16 byte stream must contain an even number of bytes.")
    low = bytearray()
    high = bytearray()
    for i in range(0, len(data), 2):
        low.append(data[i])
        high.append(data[i + 1])
    return bytes(high + low)


def byte_merge(data: bytes) -> bytes:
    if len(data) % 2 != 0:
        raise ValueError("Compressed byte stream is malformed.")
    n = len(data) // 2
    high = data[:n]
    low = data[n:]
    merged = bytearray()
    for i in range(n):
        merged.append(low[i])
        merged.append(high[i])

    return bytes(merged)


def compress(data: bytes, level: int = 3):
    transformed = byte_split(data)
    compressor = zstd.ZstdCompressor(level=level)
    start = time.perf_counter()
    compressed = compressor.compress(transformed)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def decompress(data: bytes):
    decompressor = zstd.ZstdDecompressor()
    transformed = decompressor.decompress(data)
    return byte_merge(transformed)


def compression_ratio(original_size, compressed_size):
    return compressed_size / original_size


def compression_stats(original_bytes: bytes):
    compressed, comp_time = compress(original_bytes)
    original_size = len(original_bytes)
    compressed_size = len(compressed)

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "saved_bytes": original_size - compressed_size,
        "compression_ratio": compressed_size / original_size,
        "compression_percent": 100 * (1 - compressed_size / original_size),
        "compression_time": comp_time,
    }

if __name__ == "__main__":
    import numpy as np
    activations = np.random.randn(32, 128, 5120).astype(np.float16)
    payload = activations.tobytes()
    stats = compression_stats(payload)
    print("\nBloomBee Compression")
    print("--------------------")
    print(f"Original Size     : {stats['original_size']:,} bytes")
    print(f"Compressed Size   : {stats['compressed_size']:,} bytes")
    print(f"Saved             : {stats['saved_bytes']:,} bytes")
    print(f"Compression Ratio : {stats['compression_ratio']:.3f}")
    print(f"Reduction         : {stats['compression_percent']:.2f}%")
    print(f"Compression Time  : {stats['compression_time']:.6f} sec")