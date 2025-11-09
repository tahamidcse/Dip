import numpy as np
import cv2
import matplotlib.pyplot as plt

def run_length_encode(data):
    """Run-Length Encoding for 1D array"""
    if len(data) == 0:
        return []
    
    encoded = []
    count = 1
    
    for i in range(1, len(data)):
        if data[i] == data[i-1]:
            count += 1
        else:
            encoded.extend([count, data[i-1]])
            count = 1
    
    # Add the last run
    encoded.extend([count, data[-1]])
    return encoded

def run_length_decode(encoded):
    """Run-Length Decoding for encoded array"""
    if len(encoded) == 0:
        return []
    
    decoded = []
    
    # Process pairs (count, value)
    for i in range(0, len(encoded), 2):
        if i + 1 < len(encoded):
            count = encoded[i]
            value = encoded[i + 1]
            decoded.extend([value] * count)
    
    return decoded

def rle_image_processing(image):
    """Complete RLE process for images"""
    print('The pixel values are:')
    print(image)
    
    # Display original image
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Convert 2D image to 1D array (row-wise, like MATLAB)
    str_1d = image.flatten()
    print('The input string is:')
    print(str_1d)
    
    # Run-Length Encoding
    encoded = run_length_encode(str_1d)
    print('The encoded string is:')
    print(encoded)
    
    # Calculate compression ratio
    original_size = len(str_1d)
    encoded_size = len(encoded)
    compression_ratio = original_size / encoded_size if encoded_size > 0 else 1
    
    print(f'The compression ratio is: {compression_ratio:.2f}')
    
    # Run-Length Decoding
    decoded = run_length_decode(encoded)
    print('The output string is:')
    print(decoded)
    
    # Reshape back to original image dimensions
    restored_image = np.array(decoded).reshape(image.shape)
    
    # Display restored image
    plt.subplot(1, 2, 2)
    plt.imshow(restored_image, cmap='gray')
    plt.title('Decoded Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return encoded, restored_image, compression_ratio

# Test with the same matrix from MATLAB code
def test_with_matlab_example():
    """Test with the exact same matrix from MATLAB code"""
    I = np.array([
        [39, 39, 39, 39, 39, 126, 126],
        [39, 39, 39, 39, 39, 126, 126],
        [39, 39, 39, 39, 39, 126, 126],
        [39, 39, 39, 39, 39, 126, 126],
        [39, 39, 39, 39, 39, 126, 126],
        [39, 39, 39, 39, 39, 126, 126],
        [39, 39, 39, 39, 39, 126, 126]
    ], dtype=np.uint8)
    
    encoded, restored, cr = rle_image_processing(I)
    
    # Verify the restoration
    print(f"Restoration successful: {np.array_equal(I, restored)}")

def rle_actual_image(image_path):
    """Process actual image file with RLE"""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Original image shape: {img.shape}")
    print(f"Original image size: {img.size} pixels")
    
    # Apply RLE
    encoded, restored, cr = rle_image_processing(img)
    
    print(f"Original size: {img.size} elements")
    print(f"Encoded size: {len(encoded)} elements")
    print(f"Compression ratio: {cr:.2f}")
    print(f"Space saving: {(1 - 1/cr) * 100:.1f}%")
    
    return encoded, restored, cr

def advanced_rle_image(image, use_column_major=False):
    """Advanced RLE with different scanning patterns"""
    if use_column_major:
        # Column-major order (like MATLAB's default)
        str_1d = image.flatten('F')  # Fortran-style (column-major)
        scan_type = "Column-major"
    else:
        # Row-major order (default)
        str_1d = image.flatten()  # C-style (row-major)
        scan_type = "Row-major"
    
    print(f"\n--- {scan_type} Scanning ---")
    print(f"Original 1D array: {str_1d}")
    
    encoded = run_length_encode(str_1d)
    decoded = run_length_decode(encoded)
    
    if use_column_major:
        restored = np.array(decoded).reshape(image.shape, order='F')
    else:
        restored = np.array(decoded).reshape(image.shape)
    
    compression_ratio = len(str_1d) / len(encoded) if len(encoded) > 0 else 1
    
    print(f"Encoded: {encoded}")
    print(f"Compression ratio: {compression_ratio:.2f}")
    print(f"Restoration successful: {np.array_equal(image, restored)}")
    
    return encoded, restored, compression_ratio

# Example usage
if __name__ == "__main__":
    print("=== Testing with MATLAB example matrix ===")
    test_with_matlab_example()
    
    print("\n=== Testing with different scanning patterns ===")
    test_matrix = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ], dtype=np.uint8)
    
    print("Test matrix:")
    print(test_matrix)
    
    # Row-major
    encoded_row, restored_row, cr_row = advanced_rle_image(test_matrix, False)
    
    # Column-major  
    encoded_col, restored_col, cr_col = advanced_rle_image(test_matrix, True)
    
    print(f"\nComparison:")
    print(f"Row-major compression ratio: {cr_row:.2f}")
    print(f"Column-major compression ratio: {cr_col:.2f}")
    
    # For actual image file (uncomment to use)
    # image_path = "your_image.jpg"
    # encoded, restored, cr = rle_actual_image(image_path)
