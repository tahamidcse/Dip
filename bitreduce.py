import cv2
import numpy as np

def convert_to_8_colors_bgr(image):
    """
    Convert BGR image to 8 colors (3 red, 3 green, 2 blue)
    
    Args:
        image: Input BGR image (OpenCV format)
    
    Returns:
        Converted image with 8 colors
    """
    # Make a copy of the image
    result = image.copy()
    
    # Split into B, G, R channels
    b, g, r = cv2.split(image)
    
    # Convert red channel (8-bit to 3 levels)
    # 0-85 -> 42 (level 0), 86-170 -> 128 (level 1), 171-255 -> 213 (level 2)
    r_quantized = np.zeros_like(r)
    r_quantized[r < 86] = 42
    r_quantized[(r >= 86) & (r < 171)] = 128
    r_quantized[r >= 171] = 213
    
    # Convert green channel (8-bit to 3 levels)
    # 0-85 -> 42 (level 0), 86-170 -> 128 (level 1), 171-255 -> 213 (level 2)
    g_quantized = np.zeros_like(g)
    g_quantized[g < 86] = 42
    g_quantized[(g >= 86) & (g < 171)] = 128
    g_quantized[g >= 171] = 213
    
    # Convert blue channel (8-bit to 2 levels)
    # 0-127 -> 64 (level 0), 128-255 -> 192 (level 1)
    b_quantized = np.zeros_like(b)
    b_quantized[b < 128] = 64
    b_quantized[b >= 128] = 192
    
    # Merge channels back
    result = cv2.merge([b_quantized, g_quantized, r_quantized])
    
    return result

# Alternative implementation using integer division for cleaner code
def convert_to_8_colors_bgr_v2(image):
    """
    Alternative implementation using integer division
    """
    result = image.copy()
    
    b, g, r = cv2.split(image)
    
    # Red: 3 levels (0, 1, 2) -> map to (42, 128, 213)
    r_levels = (r // 86).astype(np.uint8)  # 0-2
    r_quantized = np.where(r_levels == 0, 42, np.where(r_levels == 1, 128, 213))
    
    # Green: 3 levels (0, 1, 2) -> map to (42, 128, 213)
    g_levels = (g // 86).astype(np.uint8)  # 0-2
    g_quantized = np.where(g_levels == 0, 42, np.where(g_levels == 1, 128, 213))
    
    # Blue: 2 levels (0, 1) -> map to (64, 192)
    b_levels = (b // 128).astype(np.uint8)  # 0-1
    b_quantized = np.where(b_levels == 0, 64, 192)
    
    return cv2.merge([b_quantized, g_quantized, r_quantized])

# Example usage
if __name__ == "__main__":
    # Read input image
    image = cv2.imread('input_image.jpg')  # Replace with your image path
    
    if image is None:
        print("Error: Could not load image")
        # Create a sample image for demonstration
        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        print("Using random sample image")
    
    # Convert to 8 colors
    converted_image = convert_to_8_colors_bgr(image)
    
    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('8 Colors (3R, 3G, 2B)', converted_image)
    
    print("Original image shape:", image.shape)
    print("Original image dtype:", image.dtype)
    print("Converted image shape:", converted_image.shape)
    print("Converted image dtype:", converted_image.dtype)
    
    # Count unique colors in converted image
    unique_colors = np.unique(converted_image.reshape(-1, 3), axis=0)
    print(f"Number of unique colors in converted image: {len(unique_colors)}")
    print("Unique colors (BGR format):")
    for color in unique_colors:
        print(f"  B:{color[0]:3d}, G:{color[1]:3d}, R:{color[2]:3d}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    cv2.imwrite('converted_8_colors.jpg', converted_image)
    print("Converted image saved as 'converted_8_colors.jpg'")
