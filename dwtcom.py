
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt # Make sure to install: pip install PyWavelets
import huffman  # pip install huffman
from collections import Counter
from skimage.metrics import peak_signal_noise_ratio as psnr
import pickle

# --- ALGORITHM: DWT_Based_Image_Compression ---

# === Step 1: Load Image ===
image_path = '/content/Fig0431(d)(blown_ic_crop).tif'
f = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError(f"Image '{image_path}' not found")

original_shape = f.shape
r, c = original_shape

plt.figure(figsize=(8, 6))
plt.imshow(f, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# === Step 2: Initialize Parameters ===
L = 3  # Number of DWT decomposition levels
wavelet_type = 'haar'  # Wavelet filter (e.g., 'haar', 'db4', 'sym5')

# Simplified Quantization Scales (Frequency Weights)
quantization_scale_factors = {
    'LL': 4,
    'LH': 8,
    'HL': 8,
    'HH': 16
}

# Huffman compression/decompression functions (reused)
def huffman_compress_coefficients(coefficients):
    # Flatten and convert to integers for Huffman
    flat_coeffs = coefficients.flatten()
    # Scale and cast to int (e.g., multiplying by 10)
    coeffs_int = np.round(flat_coeffs * 10).astype(np.int32)
    
    # Build frequency dictionary
    freq_dict = Counter(coeffs_int)
    
    if not freq_dict:
        return "", {}, coefficients.nbytes * 8
        
    codebook = huffman.codebook(freq_dict.items())
    encoded_bits = ''.join(codebook[symbol] for symbol in coeffs_int)
    
    return encoded_bits, codebook, coeffs_int.nbytes * 8

def huffman_decompress_coefficients(encoded_bits, codebook, original_shape):
    reverse_codebook = {v: k for k, v in codebook.items()}
    current_code = ""
    decoded_coeffs_int = []
    
    for bit in encoded_bits:
        current_code += bit
        if current_code in reverse_codebook:
            decoded_coeffs_int.append(reverse_codebook[current_code])
            current_code = ""
    
    decoded_coeffs_float = np.array(decoded_coeffs_int, dtype=np.float32) / 10.0
    
    return decoded_coeffs_float.reshape(original_shape)

# === Step 3: Level Shifting ===
f_shifted = f.astype(np.float32) - 128
print("Level shifting applied.")

# === Step 4: Multi-level 2D DWT Decomposition ===
print(f"Applying {L}-level DWT with '{wavelet_type}' wavelet...")
# The decomposition structure: [LL_L, (LH_L, HL_L, HH_L), ..., (LH_1, HL_1, HH_1)]
coeffs = pywt.wavedec2(f_shifted, wavelet_type, level=L)
# 

# === Step 5: Coefficient Organization ===
LL_L = coeffs[0]
detail_subbands = coeffs[1:]

# === Step 6: Quantization with Frequency Weighting ===
quantized_coeffs_list = [LL_L] # Start with LL subband

for level_idx, detail_tuple in enumerate(detail_subbands):
    level = L - level_idx
    LH, HL, HH = detail_tuple
    
    print(f"Quantizing Level {level} subbands...")
    
    # Calculate quantization step (simplified with frequency weighting/level factor)
    scale_LH = quantization_scale_factors['LH'] * level 
    scale_HL = quantization_scale_factors['HL'] * level 
    scale_HH = quantization_scale_factors['HH'] * level * 1.5
    
    # APPLY uniform quantization
    LH_quantized = np.round(LH / scale_LH)
    HL_quantized = np.round(HL / scale_HL)
    HH_quantized = np.round(HH / scale_HH)
    
    # De-quantization (required for reconstruction)
    LH_dequantized = LH_quantized * scale_LH
    HL_dequantized = HL_quantized * scale_HL
    HH_dequantized = HH_quantized * scale_HH
    
    # Store for encoding and reconstruction
    quantized_coeffs_list.append((LH_quantized, HL_quantized, HH_quantized))
    detail_subbands[level_idx] = (LH_dequantized, HL_dequantized, HH_dequantized)

# Quantize LL_L approximation subband
scale_LL = quantization_scale_factors['LL'] * L 
LL_L_quantized = np.round(LL_L / scale_LL)
LL_L_dequantized = LL_L_quantized * scale_LL
quantized_coeffs_list[0] = LL_L_quantized

# === Step 7: Entropy Encoding (Huffman) ===
encoded_data = []
total_original_bits = 0
total_compressed_bits = 0

# Encode LL subband
encoded, codebook, original_size = huffman_compress_coefficients(quantized_coeffs_list[0])
encoded_data.append({'subband': 'LL_L', 'encoded_bits': encoded, 'codebook': codebook, 'shape': quantized_coeffs_list[0].shape})
total_original_bits += original_size
total_compressed_bits += len(encoded)

# Encode detail subbands
subband_names = ['LH', 'HL', 'HH']
for level_idx, detail_tuple in enumerate(quantized_coeffs_list[1:]):
    level = L - level_idx
    for i, subband_coeffs in enumerate(detail_tuple):
        name = f"{subband_names[i]}_{level}"
        encoded, codebook, original_size = huffman_compress_coefficients(subband_coeffs)
        encoded_data.append({'subband': name, 'encoded_bits': encoded, 'codebook': codebook, 'shape': subband_coeffs.shape})
        total_original_bits += original_size
        total_compressed_bits += len(encoded)

print("Quantization and Huffman Encoding Complete.")

# === Step 8: Save Compressed File ===
compressed_file_data = {
    'wavelet': wavelet_type,
    'levels': L,
    'shape': original_shape,
    'encoded_data': encoded_data,
    'quant_scales': quantization_scale_factors,
    'compression_metrics': {
        'total_original_bits': total_original_bits,
        'total_compressed_bits': total_compressed_bits
    }
}

with open('dwt_compressed_image.pkl', 'wb') as f_out:
    pickle.dump(compressed_file_data, f_out)
print(f"Compressed data saved as 'dwt_compressed_image.pkl'")

# --- DECOMPRESSION (Reconstruction) Starts Here ---

# Load and decode
decoded_coeffs = []
for item in encoded_data:
    decoded_arr = huffman_decompress_coefficients(item['encoded_bits'], item['codebook'], item['shape'])
    decoded_coeffs.append(decoded_arr)

# Reassemble the list structure for pywt.waverec2
# 1. Dequantize LL_L (index 0)
LL_L_reconstructed = decoded_coeffs[0] * (quantization_scale_factors['LL'] * L)

# 2. Dequantize and group detail subbands
reconstructed_coeffs_list = [LL_L_reconstructed]
detail_index = 1
for level in range(L, 0, -1):
    # Retrieve the scales used for quantization
    scale_LH = quantization_scale_factors['LH'] * level 
    scale_HL = quantization_scale_factors['HL'] * level 
    scale_HH = quantization_scale_factors['HH'] * level * 1.5
    
    # Dequantize
    LH_q = decoded_coeffs[detail_index]
    HL_q = decoded_coeffs[detail_index + 1]
    HH_q = decoded_coeffs[detail_index + 2]
    
    LH_deq = LH_q * scale_LH
    HL_deq = HL_q * scale_HL
    HH_deq = HH_q * scale_HH
    
    reconstructed_coeffs_list.append((LH_deq, HL_deq, HH_deq))
    detail_index += 3

print("Huffman Decoding and Dequantization Complete.")

# === Step 9: Reconstruction (Decompression) ===
# Apply inverse 2D DWT for each level
f_reconstructed_shifted = pywt.waverec2(reconstructed_coeffs_list, wavelet_type)

# Apply inverse level shifting: f_reconstructed = f + 128
f_reconstructed = f_reconstructed_shifted + 128
f_reconstructed = np.clip(f_reconstructed, 0, 255).astype(np.uint8)

print("Inverse DWT and Inverse Level Shifting Complete.")

# === Step 10: Display Results and Calculate Metrics ===
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(f, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(f_reconstructed, cmap='gray')
plt.title(f'DWT Compressed Reconstruction (L={L}, {wavelet_type})')
plt.axis('off')

plt.tight_layout()
plt.show()

# Calculate Compression Metrics
original_size_bits = r * c * 8
compression_ratio = original_size_bits / total_compressed_bits if total_compressed_bits > 0 else 0
bit_rate = total_compressed_bits / (r * c)

# PSNR (Peak Signal-to-Noise Ratio)
psnr_val = psnr(f, f_reconstructed, data_range=255)

print("\n=== DWT Compression Metrics ===")
print(f"Wavelet Type: {wavelet_type}")
print(f"Decomposition Levels (L): {L}")
print(f"Original Size: {original_size_bits} bits")
print(f"Compressed Size (Huffman): {total_compressed_bits} bits")
print(f"Compression Ratio: {compression_ratio:.2f}:1")
print(f"Effective Bit Rate: {bit_rate:.2f} bpp (bits per pixel)")
print(f"PSNR (Quality): {psnr_val:.2f} dB")

# === END ===
