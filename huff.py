

import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_contrast(image, contrast_level):
    """Adjust image contrast"""
    if contrast_level == 'low':
        alpha = 0.01  # Lower contrast
    elif contrast_level == 'high':
        alpha = 1.99  # Higher contrast
    else:  # normal
        alpha = 1.0  # Original contrast
    
    beta = 0  # Brightness adjustment
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def create_circular_mask(shape, radius):
    rows, cols = shape
    center = (cols // 2, rows // 2)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)  # -1 means filled circle
    return mask

def create_band_pass_mask(shape, inner_radius, outer_radius):
    rows, cols = shape
    center = (cols // 2, rows // 2)
    
    # Create outer circle
    outer_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(outer_mask, center, outer_radius, 1, -1)
    
    # Create inner circle (to be subtracted)
    inner_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(inner_mask, center, inner_radius, 1, -1)
    
    # Band-pass = outer circle minus inner circle
    band_mask = outer_mask - inner_mask
    return band_mask

def freq_domain_filter(img):
    fft = np.fft.fft2(img.astype(float))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)
    return log_magnitude

def apply_filter(img, mask):
    fft = np.fft.fft2(img.astype(float))
    fft_shifted = np.fft.fftshift(fft)
    filtered_fft = fft_shifted * mask
    fft_unshifted = np.fft.ifftshift(filtered_fft)
    reconstructed = np.fft.ifft2(fft_unshifted)
    reconstructed = np.abs(reconstructed)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def normalize_for_display(data):
    """Normalize data for proper display"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def main():
    # Load image in grayscale
    img = cv2.imread('/content/Fig0431(d)(blown_ic_crop).tif', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found or path is incorrect.")
        return

    print(f"Original image shape: {img.shape}")

    # Create contrast versions
    low_contrast_img = adjust_contrast(img, 'low')
    high_contrast_img = adjust_contrast(img, 'high')
    normal_contrast_img = adjust_contrast(img, 'normal')  # This should be same as original

    # Compute FFT magnitude spectra
    low_contrast_magnitude = freq_domain_filter(low_contrast_img)
    high_contrast_magnitude = freq_domain_filter(high_contrast_img)
    normal_contrast_magnitude = freq_domain_filter(img)  # Fixed variable name

    # Save normalized FFT images for visualization
    cv2.imwrite('low_contrast_fft.jpg', normalize_for_display(low_contrast_magnitude) * 255)
    cv2.imwrite('high_contrast_fft.jpg', normalize_for_display(high_contrast_magnitude) * 255)
    cv2.imwrite('normal_contrast_fft.jpg', normalize_for_display(normal_contrast_magnitude) * 255)

    h, w = img.shape

    # Define radii
    low_radius = 30
    high_radius = 60

    # Create masks
    lp_mask = create_circular_mask((h, w), low_radius)
    hp_mask = 1 - lp_mask  # Inverse (binary)
    bp_mask = create_band_pass_mask((h, w), low_radius, high_radius)

    # Apply filters to original image
    low_pass_img = apply_filter(img, lp_mask)
    high_pass_img = apply_filter(img, hp_mask)
    band_pass_img = apply_filter(img, bp_mask)

    # Apply filters to contrast variants
    low_pass_img_low_contrast = apply_filter(low_contrast_img, lp_mask)
    high_pass_img_low_contrast = apply_filter(low_contrast_img, hp_mask)
    band_pass_img_low_contrast = apply_filter(low_contrast_img, bp_mask)
    
    low_pass_img_high_contrast = apply_filter(high_contrast_img, lp_mask)
    high_pass_img_high_contrast = apply_filter(high_contrast_img, hp_mask)
    band_pass_img_high_contrast = apply_filter(high_contrast_img, bp_mask)
    
    low_pass_img_normal_contrast = apply_filter(normal_contrast_img, lp_mask)
    high_pass_img_normal_contrast = apply_filter(normal_contrast_img, hp_mask)
    band_pass_img_normal_contrast = apply_filter(normal_contrast_img, bp_mask)

    # Save filtered images
    cv2.imwrite('low_pass_img_lowc.jpg', low_pass_img_low_contrast)
    cv2.imwrite('high_pass_img_lowc.jpg', high_pass_img_low_contrast)
    cv2.imwrite('band_pass_img_lowc.jpg', band_pass_img_low_contrast)
    cv2.imwrite('low_pass_img_highc.jpg', low_pass_img_high_contrast)
    cv2.imwrite('high_pass_img_highc.jpg', high_pass_img_high_contrast)
    cv2.imwrite('band_pass_img_highc.jpg', band_pass_img_high_contrast)
    cv2.imwrite('low_pass_img_normalc.jpg', low_pass_img_normal_contrast)
    cv2.imwrite('high_pass_img_normalc.jpg', high_pass_img_normal_contrast)
    cv2.imwrite('band_pass_img_normalc.jpg', band_pass_img_normal_contrast)

    # Display results - Contrast comparison
    plt.figure(figsize=(20, 12))

    # Row 1: Original images with different contrasts
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(low_contrast_img, cmap='gray')
    plt.title("Low Contrast")
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.imshow(high_contrast_img, cmap='gray')
    plt.title("High Contrast")
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.imshow(normal_contrast_img, cmap='gray')
    plt.title("Normal Contrast")
    plt.axis('off')

    # Row 2: FFT magnitude spectra
    plt.subplot(3, 4, 5)
    plt.imshow(normal_contrast_magnitude, cmap='viridis')
    plt.title("FFT - Original")
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(low_contrast_magnitude, cmap='viridis')
    plt.title("FFT - Low Contrast")
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(high_contrast_magnitude, cmap='viridis')
    plt.title("FFT - High Contrast")
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.imshow(lp_mask, cmap='gray')
    plt.title("Low Pass Mask")
    plt.axis('off')

    # Row 3: Filtered results example (using normal contrast)
    plt.subplot(3, 4, 9)
    plt.imshow(low_pass_img_normal_contrast, cmap='gray')
    plt.title("Low Pass Filtered")
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(high_pass_img_normal_contrast, cmap='gray')
    plt.title("High Pass Filtered")
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.imshow(band_pass_img_normal_contrast, cmap='gray')
    plt.title("Band Pass Filtered")
    plt.axis('off')

    plt.subplot(3, 4, 12)
    plt.imshow(bp_mask, cmap='gray')
    plt.title("Band Pass Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("All images processed and saved successfully!")

if __name__ == "__main__":
    main()

‐-----‐-'huffman---------'x--------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------

import heapq
import numpy as np
import cv2,random

# -------------------------------
# Huffman Node Class
class Node:
    def __init__(self, pixel_value, freq):
        self.pixel_value = pixel_value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# -------------------------------
# Build Huffman Tree
def build_huffman_tree(freq_map):
    heap = [Node(pixel, freq) for pixel, freq in freq_map.items() if freq > 0]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(-1, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0] if heap else None

# -------------------------------
# Generate Huffman Codes (both directions)
def generate_codes(root, code="", huffman_codes=None, inverse_huffman_codes=None):
    if huffman_codes is None:
        huffman_codes = {}
    if inverse_huffman_codes is None:
        inverse_huffman_codes = {}

    if root is None:
        return huffman_codes, inverse_huffman_codes

    if root.left is None and root.right is None:
        huffman_codes[root.pixel_value] = code
        inverse_huffman_codes[code] = root.pixel_value
        return huffman_codes, inverse_huffman_codes

    generate_codes(root.left, code + "0", huffman_codes, inverse_huffman_codes)
    generate_codes(root.right, code + "1", huffman_codes, inverse_huffman_codes)

    return huffman_codes, inverse_huffman_codes

# -------------------------------
# Frequency Calculator
def calculate_frequencies(img):
    freq = {}
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            pixel = img[i, j]
            freq[pixel] = freq.get(pixel, 0) + 1
    return freq

# -------------------------------
# Compress Image using Huffman Codes
def compress_image(gray_img, huffman_codes):
    h, w = gray_img.shape
    compressed = [[''] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            pixel = gray_img[i][j]
            compressed[i][j] = huffman_codes[pixel]
    return compressed

# -------------------------------
# Decompress Image using Huffman Codes
def decompress_image(compressed_img, inverse_huffman_codes):
    h = len(compressed_img)
    w = len(compressed_img[0])
    decompressed = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            code = compressed_img[i][j]
            if code in inverse_huffman_codes:
                decompressed[i][j] = inverse_huffman_codes[code]
            else:
                raise ValueError(f"Invalid code '{code}' at ({i},{j})")
    return decompressed

# -------------------------------
# Calculate Compression Statistics
def calculate_compression_stats(original_img, compressed_img, huffman_codes):
    h, w = original_img.shape
    total_pixels = h * w

    # Original image: 8 bits per pixel
    original_size_bits = total_pixels * 8

    # Compressed image: variable bits per pixel (Huffman)
    compressed_size_bits = 0
    for i in range(h):
        for j in range(w):
            pixel = original_img[i, j]
            compressed_size_bits += len(huffman_codes[pixel])

    compression_ratio = compressed_size_bits / original_size_bits

    print("----- Compression Stats -----")
    print(f"Original size   : {original_size_bits} bits ({original_size_bits // 8} bytes)")
    print(f"Compressed size : {compressed_size_bits} bits ({compressed_size_bits // 8} bytes)")
    print(f"Compression ratio: {compression_ratio:.4f} ({compression_ratio*100:.2f}%)")
    print("-----------------------------")

    return original_size_bits, compressed_size_bits, compression_ratio
# -------------------------------
# Calculate Compression Statistics
def calculate_compression_stats(original_img, huffman_codes, freq_map):
    h, w = original_img.shape
    total_pixels = h * w

    # Original image: 8 bits per pixel
    original_size_bits = total_pixels * 8

    # Calculate b_avg (Average Code Length) and Compressed Size
    compressed_size_bits = 0
    b_avg = 0.0

    # L is the number of unique intensity values (symbols)
    L = len(freq_map)

    # Calculate b_avg = sum(l_k * p_r(r_k))
    for pixel_value, count in freq_map.items():
        # r_k is the pixel_value
        # p_r(r_k) is the probability: count / total_pixels
        p_r_rk = count / total_pixels

        # l_k is the length of the Huffman code
        l_k = len(huffman_codes[pixel_value])

        # Add to b_avg (weighted average)
        b_avg += l_k * p_r_rk

        # Calculate compressed size: Sum(l_k * count_k)
        compressed_size_bits += l_k * count

    # The Compression Ratio is defined as: (Compressed Size / Original Size)
    # The image compression literature often defines Compression Ratio differently,
    # but based on the provided formula context (Data Redundancy = 1 - 1/C),
    # C is likely (Original Bits / Compressed Bits), so the inverse is used here:
    compression_ratio = original_size_bits / compressed_size_bits  # Original / Compressed

    # The 'Redundancy' formula from your image is: R = 1 - 1/C
    # where C is the Compression Ratio (Original/Compressed)
    data_redundancy = 1 - (1 / compression_ratio) if compression_ratio != 0 else np.inf

    print("----- Compression Stats -----")
    print(f"Total Pixels (M x N)  : {total_pixels}")
    print(f"Original size (Fixed) : {original_size_bits} bits")
    print(f"Compressed size (Huff): {compressed_size_bits} bits")
    print(f"Average Code Length (b_avg) : {b_avg:.4f} bits/pixel")
    print(f"Compression Ratio (C) : {compression_ratio:.4f} (Original/Compressed)")
    print(f"Data Redundancy (R)   : {data_redundancy:.4f}")
    print("-----------------------------")

    return original_size_bits, compressed_size_bits, b_avg, compression_ratio
# -------------------------------
def main():
    # ... (code to load image) ...
    # Load grayscale image (CHANGE PATH)
    #img = cv2.imread('/content/Fig0809(a).tif', 0)
    # ...
        # Load grayscale image (CHANGE PATH)
    height, width = 512, 512
    total_pixels = height * width

    # Calculate number of pixels in each intensity range
    num_dark = int(0.6 * total_pixels)     # 60%
    num_mid = int(0.3 * total_pixels)      # 30%
    num_bright = total_pixels - num_dark - num_mid  # Remaining 10%

    # Prepare pixel values
    dark_pixels = [random.randint(0, 50) for _ in range(num_dark)]
    mid_pixels = [random.randint(51, 150) for _ in range(num_mid)]
    bright_pixels = [random.randint(151, 255) for _ in range(num_bright)]

    # Combine and shuffle
    all_pixels = dark_pixels + mid_pixels + bright_pixels
    random.shuffle(all_pixels)

    # Fill image using for loop
    image = np.zeros((height, width), dtype=np.uint8)
    idx = 0
    for i in range(height):
        for j in range(width):
            image[i][j] = all_pixels[idx]
            idx += 1
    img=image
    # ...

    # 1. Frequency map
    freq_map = calculate_frequencies(img)  # <-- freq_map is calculated here

    # 2. Build Huffman tree
    root = build_huffman_tree(freq_map)

    # 3. Generate Huffman codes
    huffman_codes, inverse_huffman_codes = generate_codes(root)

    # 4. Compress image
    compressed_img = compress_image(img, huffman_codes)

    # 5. Decompress image
    decompressed_img = decompress_image(compressed_img, inverse_huffman_codes)

    # 6. Calculate compression stats
    #    <-- Pass huffman_codes AND freq_map
    calculate_compression_stats(img, huffman_codes, freq_map)

    # 7. Save decompressed image
    cv2.imwrite('reconstructedraw.png', decompressed_img)
    print("Image compression and decompression completed successfully.")
    huffman_table_list = [[pixel, code] for pixel, code in sorted(huffman_codes.items())]

# Convert to 2D NumPy array of object type (since codes are strings)
    huffman_table_matrix = np.array(huffman_table_list, dtype=object)
# -------------------------------
if __name__ == "__main__":
    main()
