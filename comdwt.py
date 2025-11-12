from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import heapq
from collections import Counter

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

def analyze_coefficients(coeff_arr, thresholded_arr, keep_ratio):
    """Analyze coefficient statistics"""
    total_coeffs = coeff_arr.size
    non_zero_before = np.count_nonzero(coeff_arr)
    non_zero_after = np.count_nonzero(thresholded_arr)
    
    return {
        'total_coefficients': total_coeffs,
        'non_zero_before': non_zero_before,
        'non_zero_after': non_zero_after,
        'sparsity_ratio': non_zero_after / total_coeffs,
        'compression_ratio_coeffs': total_coeffs / non_zero_after if non_zero_after > 0 else 0
    }

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

print(f"\n{'='*60}")
print("WAVELET + HUFFMAN COMPRESSION RESULTS")
print(f"{'='*60}")
print(f"Original image: {H} x {W} = {H*W} pixels")
print(f"Original bits: {original_bits}")
print(f"Wavelet: {w}, Level: {n}, Quantization step: {qstep}")
print(f"{'='*60}")
print(f"{'Keep Ratio':<12} {'Non-zero Coeffs':<15} {'Huffman Bits':<15} {'Compression %':<15} {'Bits/Pixel':<12}")
print(f"{'-'*60}")

for keep in (0.1, 0.05, 0.01, 0.005):
    thresh = Csort[int(np.floor((1-keep) * len(Csort)))]
    ind = np.abs(coeff_arr_q) > thresh
    Cfilt = coeff_arr_q * ind  # Threshold small indices
    
    # Calculate Huffman bits for the thresholded coefficients
    huffman_bits = huff_bits(Cfilt)
    compression_percentage = 100 * (1 - huffman_bits / original_bits)
    bits_per_pixel = huffman_bits / (H * W)
    
    # Analyze coefficients
    stats = analyze_coefficients(coeff_arr_q, Cfilt, keep)
    
    print(f"{keep:<12} {stats['non_zero_after']:<15} {huffman_bits:<15} {compression_percentage:<15.1f} {bits_per_pixel:<12.3f}")
    
    # Dequantize before converting back to coefficient structure
    Cfilt_deq = Cfilt.astype(np.float32) * qstep
    coeffs_filt = pywt.array_to_coeffs(Cfilt_deq, coeff_slices, output_format='wavedec2')
    
    # Plot reconstruction
    Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
    plt.figure()
    plt.imshow(Arecon.astype('uint8'), cmap='gray')
    plt.axis('off')
    plt.title(f'keep={keep}, bits={huffman_bits}\ncompressed={compression_percentage:.1f}%, bpp={bits_per_pixel:.3f}')
    plt.show()

plt.rcParams['figure.figsize'] = [8, 8]
