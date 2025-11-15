import numpy as np
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

def apply_subband_quantization(coeffs, quantization_factors):
    """
    Apply subband-specific quantization to wavelet coefficients
    
    Parameters:
    - coeffs: Wavelet coefficients from pywt.wavedec2
    - quantization_factors: Dictionary with 'LL', 'LH', 'HL', 'HH' keys
    
    Returns:
    - quantized_coeffs: Quantized coefficients in array format
    - coeff_slices: Coefficient slices for reconstruction
    """
    # Convert coefficients to array format
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # Create quantization mask with subband-specific factors
    quant_mask = np.ones_like(coeff_arr, dtype=np.float32)
    
    # Get the approximation (LL) subband - it's the first element in coeffs
    LL_shape = coeffs[0].shape
    LL_size = LL_shape[0] * LL_shape[1]
    
    # Apply LL quantization factor to approximation coefficients
    # In the coefficient array, LL is at the beginning
    quant_mask[:LL_size] = quantization_factors['LL']
    
    # Apply quantization factors to detail subbands
    current_pos = LL_size
    
    # Process each level of detail coefficients
    for level in range(1, len(coeffs)):
        LH, HL, HH = coeffs[level]
        
        # Apply quantization factors to each detail subband
        LH_size = LH.size
        HL_size = HL.size  
        HH_size = HH.size
        
        # LH subband
        quant_mask[current_pos:current_pos + LH_size] = quantization_factors['LH']
        current_pos += LH_size
        
        # HL subband
        quant_mask[current_pos:current_pos + HL_size] = quantization_factors['HL']
        current_pos += HL_size
        
        # HH subband
        quant_mask[current_pos:current_pos + HH_size] = quantization_factors['HH']
        current_pos += HH_size
    
    # Apply quantization
    coeff_arr_quantized = np.rint(coeff_arr / quant_mask).astype(np.int32)
    
    return coeff_arr_quantized, coeff_slices, quant_mask

def dequantize_subbands(coeff_arr_quantized, quant_mask):
    """Dequantize coefficients using subband-specific factors"""
    return coeff_arr_quantized.astype(np.float32) * quant_mask

plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

# Load image
A = imread(os.path.join('..','DATA','dog.jpg'))
B = np.mean(A, -1)  # Convert RGB to grayscale

## Wavelet Compression
w = 'db1'  # Wavelet type
n = 4      # Decomposition level

# NEW: Subband-specific quantization factors
quantization_factors = {
    'LL': 2,   # Small weight → retain quality (coefficient / 2)
    'LH': 4,   # Medium weight → reduce detail (coefficient / 4)
    'HL': 4,   # Medium weight → reduce detail (coefficient / 4)
    'HH': 8    # Large weight → suppress high-frequency/noise (coefficient / 8)
}

print("Subband Quantization Factors:")
print(f"  LL (Approximation): /{quantization_factors['LL']} → preserve quality")
print(f"  LH (Horizontal): /{quantization_factors['LH']} → reduce vertical edge details")  
print(f"  HL (Vertical): /{quantization_factors['HL']} → reduce horizontal edge details")
print(f"  HH (Diagonal): /{quantization_factors['HH']} → suppress noise & fine details")
print()

# Original image size for bit calculation
H, W = B.shape
original_bits = H * W * 8  # 8 bits per pixel

# Perform wavelet decomposition
coeffs = pywt.wavedec2(B, wavelet=w, level=n)

# NEW: Apply subband-specific quantization
coeff_arr_q, coeff_slices, quant_mask = apply_subband_quantization(coeffs, quantization_factors)

Csort = np.sort(np.abs(coeff_arr_q.reshape(-1)))

print(f"\n{'='*80}")
print("WAVELET + SUBBAND QUANTIZATION + RLE + HUFFMAN COMPRESSION RESULTS")
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
    
    # NEW: Use subband-specific dequantization
    Cfilt_deq = dequantize_subbands(Cfilt, quant_mask)
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
print(f"Wavelet: {w}, Level: {n}")
print(f"Subband Quantization: LL/{quantization_factors['LL']}, LH/{quantization_factors['LH']}, HL/{quantization_factors['HL']}, HH/{quantization_factors['HH']}")
print(f"{'='*80}")
