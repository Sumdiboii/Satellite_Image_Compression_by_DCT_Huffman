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

def compress_huffman(arr):
    flat = arr.flatten()
    freq_table = dict(Counter(flat))
    root = build_tree(freq_table)
    codes = generate_codes(root)
    encoded = "".join([codes[p] for p in flat])
    return encoded, codes

def decompress_huffman(encoded, codes, shape):
    reverse_codes = {v: k for k, v in codes.items()}
    decoded = []
    current = ""
    for bit in encoded:
        current += bit
        if current in reverse_codes:
            decoded.append(reverse_codes[current])
            current = ""
    return np.array(decoded).reshape(shape)
