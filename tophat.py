import cv2
import numpy as np

def erode(input_img, kernel):
    """
    Performs a custom morphological erosion operation with padding.
    
    Args:
        input_img (np.ndarray): The binary input image (0 or 255).
        kernel (np.ndarray): The binary structuring element (0 or 255).
    
    Returns:
        np.ndarray: The resulting eroded image (0 or 255) of the same size as the input.
    """
    input_h, input_w = input_img.shape
    kernel_h, kernel_w = kernel.shape
    
    # Pad the input image to get an output of the same size
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_img = cv2.copyMakeBorder(input_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    # Normalize inputs to 0-1 for correct mathematical operations
    input_normalized = (padded_img > 0).astype(np.uint8)
    kernel_normalized = (kernel > 0).astype(np.uint8)
    
    # Initialize the output image
    output_img = np.zeros_like(input_img, dtype=np.uint8)
    
    # Loop through the padded image
    for h in range(input_h):
        for w in range(input_w):
            roi = input_normalized[h:h + kernel_h, w:w + kernel_w]
            if np.all(roi == kernel_normalized):
                output_img[h, w] = 255
    
    return output_img

def dilate(input_img, kernel):
    """
    Performs a custom morphological dilation operation with padding.
    
    Args:
        input_img (np.ndarray): The binary input image (0 or 255).
        kernel (np.ndarray): The binary structuring element (0 or 255).
    
    Returns:
        np.ndarray: The resulting dilated image (0 or 255) of the same size as the input.
    """
    input_h, input_w = input_img.shape
    kernel_h, kernel_w = kernel.shape
    
    # Pad the input image
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_img = cv2.copyMakeBorder(input_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    # Initialize the output image
    output_img = np.zeros_like(input_img, dtype=np.uint8)
    
    # Normalize inputs to 0-1 for correct mathematical operations
    input_normalized = (padded_img > 0).astype(np.uint8)
    kernel_normalized = (kernel > 0).astype(np.uint8)
    
    # Loop through the padded image
    for h in range(input_h):
        for w in range(input_w):
            roi = input_normalized[h:h + kernel_h, w:w + kernel_w]
            if np.sum(roi * kernel_normalized) > 0:
                output_img[h, w] = 255
    
    return output_img

def get_diamond_structuring_element(size):
    """
    Creates a diamond-shaped structuring element.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            if abs(i - center) + abs(j - center) <= center:
                kernel[i, j] = 1
                
    return kernel

if __name__ == '__main__':
    # Load an image or create a synthetic one if not found
    try:
        img = cv2.imread('morph.png', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    except FileNotFoundError:
        print("Image 'morph.png' not found. Creating a synthetic image.")
        img = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), 255, -1)
        cv2.circle(img, (200, 200), 5, 255, -1)
        cv2.rectangle(img, (120, 120), (130, 130), 0, -1)
    
    # Define a kernel size
    ksize = (5, 5)

    # Define the four different structuring elements
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    kernel_diamond = get_diamond_structuring_element(ksize[0])

    # Perform Black Hat transform for each kernel
    # Built-in functions for rectangular, elliptical, and cross-shaped kernels
    closed_rect = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_rect)
    blackhat_rect = closed_rect - img

    closed_ellipse = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_ellipse)
    blackhat_ellipse = closed_ellipse - img

    closed_cross = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_cross)
    blackhat_cross = closed_cross - img

    # Use the custom functions for the diamond kernel
    dilated_diamond = dilate(img, kernel_diamond)
    closed_diamond = erode(dilated_diamond, kernel_diamond)
    blackhat_diamond = closed_diamond - img
    
    # Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Black Hat (Rectangular)', blackhat_rect)
    cv2.imshow('Black Hat (Elliptical)', blackhat_ellipse)
    cv2.imshow('Black Hat (Cross-shaped)', blackhat_cross)
    cv2.imshow('Black Hat (Diamond-shaped)', blackhat_diamond)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
