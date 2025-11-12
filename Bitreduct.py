import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def compress_by_reducing_color_depth_24bit_to_8bit(img):
    """
    Image Compression by Reducing Color Depth
    24-bit RGB → 8-bit with 3R-3G-2B allocation
    """
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # Color depth reduction
    r_3bit = (r // 32).astype(np.uint8)  # 3 bits for Red
    g_3bit = (g // 32).astype(np.uint8)  # 3 bits for Green  
    b_2bit = (b // 64).astype(np.uint8)  # 2 bits for Blue
    
    # Scale back for display
    r_display = r_3bit * 36
    g_display = g_3bit * 36
    b_display = b_2bit * 85
    
    compressed_img = np.stack([r_display, g_display, b_display], axis=2)
    return compressed_img

def compress_by_reducing_color_depth_24bit_to_4bit(img):
    """
    Image Compression by Reducing Color Depth
    24-bit RGB → 4-bit with 1G-2R-1B allocation
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r, b = img[:,:,0], img[:,:,2]
    
    # Color depth reduction
    gray_1bit = (gray > 128).astype(np.uint8)  # 1 bit Gray
    r_2bit = (r // 64).astype(np.uint8)        # 2 bits Red
    b_1bit = (b > 128).astype(np.uint8)        # 1 bit Blue
    
    # Reconstruct compressed image
    compressed_img = np.zeros_like(img)
    compressed_img[:,:,0] = np.clip(gray_1bit * 255 + r_2bit * 85, 0, 255)
    compressed_img[:,:,1] = np.clip(gray_1bit * 255 * 0.7, 0, 255)
    compressed_img[:,:,2] = np.clip(gray_1bit * 255 * 0.5 + b_1bit * 128, 0, 255)
    
    return compressed_img

def calculate_compression_ratio(original_bits, compressed_bits):
    """Calculate compression ratio and percentage"""
    compression_ratio = original_bits / compressed_bits
    compression_percentage = (1 - compressed_bits / original_bits) * 100
    return compression_ratio, compression_percentage

# Main analysis function
def analyze_color_depth_compression(img):
    H, W = img.shape[:2]
    
    print("IMAGE COMPRESSION BY REDUCING COLOR DEPTH")
    print("=" * 60)
    
    # Original 24-bit
    original_size = H * W * 3  # bytes
    original_bpp = 24
    
    # 8-bit compression
    compressed_8bit = compress_by_reducing_color_depth_24bit_to_8bit(img)
    compressed_8bit_size = H * W * 1  # bytes
    ratio_8bit, pct_8bit = calculate_compression_ratio(original_bpp, 8)
    
    # 4-bit compression  
    compressed_4bit = compress_by_reducing_color_depth_24bit_to_4bit(img)
    compressed_4bit_size = H * W * 0.5  # bytes
    ratio_4bit, pct_4bit = calculate_compression_ratio(original_bpp, 4)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(img)
    axes[0].set_title(f'Original 24-bit\n{original_size:,} bytes\n{original_bpp} bpp')
    axes[0].axis('off')
    
    # 8-bit compressed
    axes[1].imshow(compressed_8bit)
    axes[1].set_title(f'8-bit Compressed\n{compressed_8bit_size:,} bytes\nCompression: {pct_8bit:.1f}%')
    axes[1].axis('off')
    
    # 4-bit compressed
    axes[2].imshow(compressed_4bit)
    axes[2].set_title(f'4-bit Compressed\n{compressed_4bit_size:,} bytes\nCompression: {pct_4bit:.1f}%')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print(f"\nCOMPRESSION ANALYSIS:")
    print(f"{'Method':<25} {'Bits/Pixel':<12} {'Colors':<15} {'Size':<12} {'Compression %':<15}")
    print("-" * 80)
    print(f"{'Original 24-bit RGB':<25} {'24 bpp':<12} {'16.7M':<15} {original_size:<12,} {'0%':<15}")
    print(f"{'8-bit (3R-3G-2B)':<25} {'8 bpp':<12} {'256':<15} {compressed_8bit_size:<12,} {pct_8bit:<15.1f}")
    print(f"{'4-bit (1G-2R-1B)':<25} {'4 bpp':<12} {'16':<15} {compressed_4bit_size:<12,} {pct_4bit:<15.1f}")
    
    print(f"\nCOMPRESSION RATIOS:")
    print(f"8-bit: {ratio_8bit:.1f}:1 reduction")
    print(f"4-bit: {ratio_4bit:.1f}:1 reduction")

# Load and process image
img = load_image()  # Your existing load function
analyze_color_depth_compression(img)
