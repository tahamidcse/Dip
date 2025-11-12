from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import heapq
from collections import Counter

def run_length_encode(arr):
    """Run-length encoding of an array"""
    if len(arr) == 0:
        return []
    
    encoded = []
    count = 1
    current = arr[0]
    
    for i in range(1, len(arr)):
        if arr[i] == current:
            count += 1
        else:
            encoded.append((current, count))
            current = arr[i]
            count = 1
    
    encoded.append((current, count))
    return encoded

def zigzag_scan(matrix):
    """Zigzag scan of a 2D matrix"""
    rows, cols = matrix.shape
    result = []
    
    # Directions: right, down-left, down, up-right
    directions = [(0, 1), (1, -1), (1, 0), (-1, 1)]
    dir_idx = 0
    i, j = 0, 0
    
    for _ in range(rows * cols):
        result.append(matrix[i, j])
        
        if dir_idx == 0:  # Moving right
            if i == 0:
                dir_idx = 1  # Go down-left
            else:
                dir_idx = 3  # Go up-right
        elif dir_idx == 1:  # Moving down-left
            if i == rows - 1 or j == 0:
                if j == 0:
                    dir_idx = 2  # Go down
                else:
                    dir_idx = 0  # Go right
        elif dir_idx == 2:  # Moving down
            if j == 0:
                dir_idx = 3  # Go up-right
            else:
                dir_idx = 1  # Go down-left
        elif dir_idx == 3:  # Moving up-right
            if j == cols - 1:
                dir_idx = 2  # Go down
            elif i == 0:
                dir_idx = 0  # Go right
        
        # Move to next position based on current direction
        di, dj = directions[dir_idx]
        i += di
        j += dj
    
    return np.array(result)

def huff_bits(arr: np.ndarray) -> int:
    flat = arr.ravel().tolist()
    if not flat: return 0
    freq = Counter(flat)
    h = []
    uid = 0
    for s, f in freq.items():
        heapq.heappush(h, (f, uid, (s, None, None)))
        uid += 1
    while len(h) > 1:
        f1, _, n1 = heapq.heappop(h)
        f2, _, n2 = heapq.heappop(h)
        heapq.heappush(h, (f1 + f2, uid, (None, n1, n2)))
        uid += 1
    root = h[0][2]
    codes = {}
    def walk(node, p=""):
        s, l, r = node
        if s is not None: 
            codes[s] = p or "0"
            return
        walk(l, p + "0")
        walk(r, p + "1")
    walk(root)
    return sum(len(codes[v]) for v in flat)

def calculate_rle_huffman_bits(coeff_arr):
    """Calculate total bits using RLE + Huffman"""
    # Apply zigzag scan to create 1D array
    zigzag_coeffs = zigzag_scan(coeff_arr)
    
    # Apply run-length encoding
    rle_pairs = run_length_encode(zigzag_coeffs)
    
    if not rle_pairs:
        return 0
    
    # Convert RLE pairs to flat array: [value1, run1, value2, run2, ...]
    flat_rle = []
    for value, run in rle_pairs:
        flat_rle.extend([value, run])
    
    # Calculate Huffman bits for the RLE data
    return huff_bits(np.array(flat_rle))

plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

# Load image
A = imread(os.path.join('..','DATA','dog.jpg'))
B = np.mean(A, -1)  # Convert RGB to grayscale

## Wavelet Compression
w = 'db1'  # Wavelet type
n = 4      # Decomposition level
qstep = 10.0  # Quantization step size

# Original image size for bit calculation
H, W = B.shape
original_bits = H * W * 8  # 8 bits per pixel

coeffs = pywt.wavedec2(B, wavelet=w, level=n)
coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

# Apply quantization to the entire coefficient array
coeff_arr_q = np.rint(coeff_arr / qstep).astype(np.int32)

Csort = np.sort(np.abs(coeff_arr_q.reshape(-1)))

print(f"\n{'='*80}")
print("WAVELET + RLE + HUFFMAN COMPRESSION RESULTS")
print(f"{'='*80}")
print(f"{'Keep Ratio':<12} {'Non-zero':<10} {'Huffman Only':<15} {'RLE+Huffman':<15} {'Compression %':<15} {'Gain %':<10}")
print(f"{'-'*80}")

for keep in (0.1, 0.05, 0.01, 0.005):
    thresh = Csort[int(np.floor((1-keep) * len(Csort)))]
    ind = np.abs(coeff_arr_q) > thresh
    Cfilt = coeff_arr_q * ind  # Threshold small indices
    
    # Calculate bits using different methods
    huffman_only_bits = huff_bits(Cfilt)
    rle_huffman_bits = calculate_rle_huffman_bits(Cfilt)
    
    # Calculate compression percentages
    compression_pct_huffman = 100 * (1 - huffman_only_bits / original_bits)
    compression_pct_rle_huffman = 100 * (1 - rle_huffman_bits / original_bits)
    
    # Calculate improvement from RLE
    gain_pct = compression_pct_rle_huffman - compression_pct_huffman
    
    non_zero_count = np.count_nonzero(Cfilt)
    
    print(f"{keep:<12} {non_zero_count:<10} {huffman_only_bits:<15} {rle_huffman_bits:<15} {compression_pct_rle_huffman:<15.1f} {gain_pct:<10.1f}")
    
    # Dequantize before converting back to coefficient structure
    Cfilt_deq = Cfilt.astype(np.float32) * qstep
    coeffs_filt = pywt.array_to_coeffs(Cfilt_deq, coeff_slices, output_format='wavedec2')
    
    # Plot reconstruction
    Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
    plt.figure()
    plt.imshow(Arecon.astype('uint8'), cmap='gray')
    plt.axis('off')
    plt.title(f'keep={keep}, RLE+Huffman bits={rle_huffman_bits}\ncompressed={compression_pct_rle_huffman:.1f}% (gain: {gain_pct:.1f}%)')
    plt.show()

plt.rcParams['figure.figsize'] = [8, 8]

# Print summary
print(f"\n{'='*80}")
print("COMPRESSION SUMMARY")
print(f"{'='*80}")
print(f"Original image size: {H} x {W} = {H*W} pixels")
print(f"Original bits: {original_bits}")
print(f"Wavelet: {w}, Level: {n}, Quantization step: {qstep}")
print(f"{'='*80}")
