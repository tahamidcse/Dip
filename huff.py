import heapq
import numpy as np
import cv2

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
# Main Program
def main():
    # ... (code to load image) ...
    # Load grayscale image (CHANGE PATH)
    img = cv2.imread('/content/Fig0809(a).tif', 0)
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
    cv2.imwrite('reconstructed.png', decompressed_img)
    print("Image compression and decompression completed successfully.")

# -------------------------------
if __name__ == "__main__":
    main()
