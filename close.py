import cv2
import numpy as np

def erode(input_img, kernel):
    """
    Performs a custom morphological fit (erosion) operation.

    Args:
        input_img (np.ndarray): The binary input image (0 or 255).
        kernel (np.ndarray): The binary structuring element (0 or 255).

    Returns:
        np.ndarray: The resulting eroded image (0 or 255).
    """
    input_h, input_w = input_img.shape
    kernel_h, kernel_w = kernel.shape

    # Normalize inputs to 0-1 for correct mathematical operations
    input_normalized = (input_img > 0).astype(np.uint8)
    kernel_normalized = (kernel > 0).astype(np.uint8)

    # Output dimensions for 'valid' operation (no padding)
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1

    # Initialize the output image with zeros
    output_img = np.zeros((output_h, output_w), dtype=np.uint8)

    # Loop through the input image to apply the kernel
    for h in range(output_h):
        for w in range(output_w):
            # Extract the region of interest (ROI) from the input image
            roi = input_normalized[h:h + kernel_h, w:w + kernel_w]

            # Check for a "fit"
            # A fit occurs if the kernel is a subset of the ROI.
            if np.all(roi == kernel_normalized):
                output_img[h, w] = 255  # Set to 255 if the kernel fits

    return output_img

def dilate(input_img, kernel):
    """
    Performs a custom morphological hit (dilation) operation.

    Args:
        input_img (np.ndarray): The binary input image (0 or 255).
        kernel (np.ndarray): The binary structuring element (0 or 255).

    Returns:
        np.ndarray: The resulting dilated image (0 or 255).
    """
    input_h, input_w = input_img.shape
    kernel_h, kernel_w = kernel.shape
    
    # Pad the input image to handle border pixels
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_img = cv2.copyMakeBorder(input_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

    # Output dimensions for 'same' operation
    output_img = np.zeros_like(input_img, dtype=np.uint8)

    # Normalize inputs to 0-1 for correct mathematical operations
    input_normalized = (padded_img > 0).astype(np.uint8)
    kernel_normalized = (kernel > 0).astype(np.uint8)

    # Loop through the padded image
    for h in range(input_h):
        for w in range(input_w):
            # Extract the region of interest (ROI) from the padded image
            roi = input_normalized[h:h + kernel_h, w:w + kernel_w]

            # Check for a "hit" (Dilation): The ROI must overlap with the kernel.
            if np.sum(roi * kernel_normalized) > 0:
                output_img[h, w] = 255

    return output_img

def get_diamond_structuring_element(ksize):
    """
    Generates a diamond-shaped structuring element.

    Args:
        ksize (int): The size of the square bounding box for the diamond.

    Returns:
        np.ndarray: The binary diamond-shaped kernel.
    """
    kernel = np.zeros((ksize, ksize), np.uint8)
    center = ksize // 2
    for i in range(ksize):
        for j in range(ksize):
            if abs(i - center) + abs(j - center) <= center:
                kernel[i, j] = 1
    return kernel

# Load an image (e.g., a binary or grayscale image)
# Make sure you have an image named 'morph.png' in the same directory.
try:
    img = cv2.imread('morph.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image file 'morph.png' not found.")
except FileNotFoundError as e:
    print(e)
    # Create a dummy image for demonstration if the file is not found
    print("Creating a dummy image for demonstration.")
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    cv2.circle(img, (100, 100), 20, 0, -1)

# Define a kernel size
ksize = (15, 15)

# --- Rectangular Kernel ---
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
closed_image_rect = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_rect)
cv2.imshow('Rectangular Kernel', closed_image_rect)

# --- Elliptical Kernel ---
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
closed_image_ellipse = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_ellipse)
cv2.imshow('Elliptical Kernel', closed_image_ellipse)

# --- Cross-shaped Kernel ---
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
closed_image_cross = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_cross)
cv2.imshow('Cross-shaped Kernel', closed_image_cross)

# --- Diamond-shaped Kernel ---
kernel_diamond = get_diamond_structuring_element(ksize[0])
closed_image_diamond = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_diamond)
cv2.imshow('Diamond-shaped Kernel', closed_image_diamond)

# --- Custom Close Operation (Erode + Dilate) with a Rectangular Kernel ---
eroded_image_custom = dilate(img, kernel_rect) # Dilation first for closing
closed_image_custom = erode(eroded_image_custom, kernel_rect)
cv2.imshow('Closed Image (Custom Dilate + Erode)', closed_image_custom)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
