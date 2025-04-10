import numpy as np
from collections import Counter
import heapq

class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# --------------------------
# Build Tree & Generate Codes
# --------------------------
def build_tree(freq_table):
    heap = [Node(symbol, freq) for symbol, freq in freq_table.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(None, n1.freq + n2.freq)
        merged.left, merged.right = n1, n2
        heapq.heappush(heap, merged)
    return heap[0]

def generate_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        generate_codes(node.left, prefix + "0", codebook)
        generate_codes(node.right, prefix + "1", codebook)
    return codebook

# --------------------------
# Compress
# --------------------------
def compress_huffman(arr):
    arr = np.round(arr).astype(np.int16)  # ensure integer format for symbols
    flat = arr.flatten()
    freq_table = dict(Counter(flat))
    root = build_tree(freq_table)
    codes = generate_codes(root)
    
    encoded = "".join([codes[p] for p in flat])

    # Convert bitstring to actual bytes
    byte_array = bytearray()
    for i in range(0, len(encoded), 8):
        byte = encoded[i:i+8]
        if len(byte) < 8:
            byte = byte.ljust(8, '0')  # pad the last byte
        byte_array.append(int(byte, 2))

    metadata = {
        'shape': arr.shape,
        'codes': codes,
        'dtype': arr.dtype.name  # store dtype info (e.g., 'int16')
    }

    return byte_array, metadata

# --------------------------
# Decompress
# --------------------------
def decompress_huffman(byte_array, metadata):
    bitstring = "".join(f"{byte:08b}" for byte in byte_array)
    reverse_codes = {v: k for k, v in metadata['codes'].items()}

    decoded = []
    current = ""
    expected_length = np.prod(metadata['shape'])  # total number of values expected

    for bit in bitstring:
        current += bit
        if current in reverse_codes:
            decoded.append(reverse_codes[current])
            current = ""

            # Stop once we decode enough values
            if len(decoded) == expected_length:
                break

    arr = np.array(decoded, dtype=np.dtype(metadata['dtype']))
    return arr.reshape(metadata['shape'])

