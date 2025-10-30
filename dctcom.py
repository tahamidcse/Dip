
import numpy as np
import cv2
import matplotlib.pyplot as plt
import huffman  
from collections import Counter
from skimage.metrics import peak_signal_noise_ratio as psnr # Added for metrics
import pickle # Added for saving data

# === START ===

# === Load Image (f) ===
# NOTE: The path '/content/Fig0431(d)(blown_ic_crop).tif' suggests a Google Colab environment.
image_path = '/content/Fig0431(d)(blown_ic_crop).tif'
x = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if x is None:
    # If the specific path fails, try a generic image for testing, or re-raise with correct name
    try:
        x = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
        if x is None:
             raise FileNotFoundError(f"Image not found at original path or 'cameraman.tif'")
    except FileNotFoundError as e:
        print(e)
        # Assuming the original image loaded successfully for the rest of the execution
        pass 

# Ensure x is loaded (assuming it loaded successfully if this point is reached, 
# or using a loaded x from an interactive session)

r, c = x.shape
print(f"Loaded image with shape: {r}x{c}")

plt.figure(figsize=(8, 6))
plt.imshow(x, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# === Initialize Quantization Depths, M, N, Ψ, ξ ===
depth = 4 # For truncation
N = 8 # Block size

# Initialize arrays
DF = np.zeros((r, c), dtype=np.float32)
DFF = np.zeros((r, c), dtype=np.float32)
IDF = np.zeros((r, c), dtype=np.float32)
IDFF = np.zeros((r, c), dtype=np.float32)

# Quantization matrix (JPEG standard)
quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Huffman compression function
def huffman_compress_image_coefficients(coefficients):
    # Flatten and convert to integers
    flat_coeffs = coefficients.flatten()
    coeffs_int = [int(x * 1000) for x in flat_coeffs] 
    
    freq_dict = Counter(coeffs_int)
    if not freq_dict: return "", {}, 0
    codebook = huffman.codebook(freq_dict.items())
    encoded_bits = ''.join(codebook[symbol] for symbol in coeffs_int)
    
    # Return original size in bits based on the numpy array size
    return encoded_bits, codebook, coefficients.nbytes * 8 

# Huffman decompression function (Adjusted to return numpy array)
def huffman_decompress_image_coefficients(encoded_bits, codebook, original_shape):
    reverse_codebook = {v: k for k, v in codebook.items()}
    current_code = ""
    decoded_coeffs = []
    
    for bit in encoded_bits:
        current_code += bit
        if current_code in reverse_codebook:
            decoded_coeffs.append(reverse_codebook[current_code])
            current_code = ""
    
    decoded_coeffs_float = np.array(decoded_coeffs, dtype=np.float32) / 1000.0
    return decoded_coeffs_float.reshape(original_shape)

# Store Huffman data for analysis
huffman_data = []
total_original_bits = 0
total_compressed_bits = 0

# === Loop (m, n) With inc. 8 ===
for i in range(0, r, N):
    for j in range(0, c, N):
        # Extract 8x8 block boundaries
        end_i = min(i + N, r)
        end_j = min(j + N, c)
        
        # Get block
        block = x[i:end_i, j:end_j].astype(np.float32)
        
        # Store original block shape for reconstruction assignment
        original_block_r, original_block_c = block.shape

        # Padding (if necessary) to ensure N x N for DCT
        if block.shape != (N, N):
            padded_block = np.zeros((N, N), dtype=np.float32)
            padded_block[:block.shape[0], :block.shape[1]] = block
            block = padded_block
        
        # === DCT Calculation ===
        df = cv2.dct(block)
        
        # Store DCT coefficients (only the part corresponding to the original image slice)
        # This is where the correction is applied: use end_i/end_j for slicing DF
        DF[i:end_i, j:end_j] = df[:original_block_r, :original_block_c]
        
        # === Inverse DCT using cv2.idct() to verify ===
        dff = cv2.idct(df)
        DFF[i:end_i, j:end_j] = dff[:original_block_r, :original_block_c]
        
        # === Quantize and De-quantize ===
        # Apply quantization to the full 8x8 DCT block
        df_quantized = np.round(df / quantization_matrix)
        df_dequantized = df_quantized * quantization_matrix
        
        # Additional frequency truncation (on the full 8x8 block)
        df_dequantized[depth:, :] = 0
        df_dequantized[:, depth:] = 0
        
        # Store De-quantized coefficients
        IDF[i:end_i, j:end_j] = df_dequantized[:original_block_r, :original_block_c]
        
        # === Encode (Huffman) ===
        huffman_encoded, huffman_codebook, original_size = huffman_compress_image_coefficients(df_quantized)
        
        total_original_bits += original_size
        total_compressed_bits += len(huffman_encoded)
        
        # Store Huffman data for this block
        huffman_data.append({
            'block_position': (i, j),
            'encoded_bits': huffman_encoded,
            'codebook': huffman_codebook,
            'original_shape': df_quantized.shape,
            'original_size': original_size,
            'compressed_size': len(huffman_encoded)
        })
        
        # Reconstruct from quantized coefficients using cv2.idct()
        dff_compressed = cv2.idct(df_dequantized)
        
        # Assign back the original slice size
        IDFF[i:end_i, j:end_j] = dff_compressed[:original_block_r, :original_block_c]

# --- Post-Processing and Display (Unchanged) ---

# === Display Results === 
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(DF, cmap='gray')
plt.title('DCT Coefficients')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(DFF * 0.005, cmap='gray')
plt.title('Reconstructed from Full DCT')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(IDF, cmap='gray')
plt.title('Quantized DCT Coefficients')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(IDFF * 0.004, cmap='gray')
plt.title('Compressed Reconstruction')
plt.axis('off')

plt.tight_layout()
plt.show()


# Save reconstructed image
cv2.imwrite('compressed_reconstruction.jpg', np.clip(IDFF, 0, 255).astype(np.uint8))
print("Reconstructed image saved as 'compressed_reconstruction.jpg'")

# Calculate compression statistics
original_size = r * c * 8  # bits
compressed_size_estimate = total_compressed_bits

print(f'\n=== Overall Compression Results ===')
print(f'Original image size: {original_size} bits')
print(f'Compressed size (with Huffman): {compressed_size_estimate} bits')
print(f'Overall compression ratio: {compression_ratio:.2f}:1')
print(f'Effective bit rate: {compressed_size_estimate/(r*c):.2f} bpp')

# Calculate PSNR (using IDFF, the compressed reconstruction)
mse_compressed = np.mean((x.astype(np.float32) - IDFF) ** 2)

if mse_compressed > 0:
    psnr_compressed = 20 * np.log10(255.0 / np.sqrt(mse_compressed))
else:
    psnr_compressed = float('inf')

print(f'PSNR (Compressed): {psnr_compressed:.2f} dB')
