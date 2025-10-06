import numpy as np
from collections import Counter
from bitarray import bitarray
from heapq import heappush, heappop

class HuffNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [HuffNode(sym, freq) for sym, freq in freq_dict.items()]
    for node in heap:
        heappush(heap, node)
    while len(heap) > 1:
        n1 = heappop(heap)
        n2 = heappop(heap)
        merged = HuffNode(None, n1.freq + n2.freq, n1, n2)
        heappush(heap, merged)
    return heap[0]

def generate_codes(node, prefix='', codebook={}):
    if node is None:
        return
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    generate_codes(node.left, prefix + '0', codebook)
    generate_codes(node.right, prefix + '1', codebook)
    return codebook

def mat2huff(x):
    # Check input
    if not isinstance(x, (np.ndarray, list)):
        raise ValueError("Input must be a 2D array or list")
    
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Input must be 2D")

    # Convert to integers
    x = np.round(x).astype(int)
    xmin = x.min()
    xmax = x.max()
    
    # Bias the minimum by 32768
    pmin = int(xmin) + 32768
    y_min = np.uint16(pmin)

    # Histogram
    flat_x = x.flatten()
    symbols = flat_x - xmin  # shift to zero-based indexing
    counter = Counter(symbols)
    
    # Limit max histogram values to 65535
    max_freq = max(counter.values())
    if max_freq > 65535:
        scale = 65535 / max_freq
        for k in counter:
            counter[k] = int(counter[k] * scale)

    hist = np.zeros(xmax - xmin + 1, dtype=np.uint16)
    for sym, freq in counter.items():
        hist[sym] = freq

    # Build Huffman codebook
    huff_tree = build_huffman_tree(counter)
    huff_map = generate_codes(huff_tree)

    # Encode the data
    encoded_bits = ''.join(huff_map[sym] for sym in symbols)
    
    # Pad to 16-bit alignment
    pad_len = (16 - len(encoded_bits) % 16) % 16
    encoded_bits += '0' * pad_len

    # Convert to uint16 values
    words = [encoded_bits[i:i+16] for i in range(0, len(encoded_bits), 16)]
    uint16_vals = [int(word, 2) for word in words]
    code = np.array(uint16_vals, dtype=np.uint16)

    # Output structure
    y = {
        'code': code,
        'min': y_min,
        'size': np.array(x.shape, dtype=np.uint32),
        'hist': hist
    }

    return y
# Example matrix
img = np.array([[3, 3, 2], [1, 3, 2]])

# Compress
y = mat2huff(img)

print("Encoded Code:", y['code'])
print("Min Value + 32768:", y['min'])
print("Original Size:", y['size'])
print("Histogram:", y['hist'])
