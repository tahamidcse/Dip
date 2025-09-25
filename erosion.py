import numpy as np
import cv2
from skimage.morphology import diamond
def morphological_fit(input_img, kernel):
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



if __name__ == '__main__':
    # Load the specified image. If not found, a synthetic one is created.
    try:
        input_image = cv2.imread('FigP0919(UTK).tif', cv2.IMREAD_GRAYSCALE)
        if input_image is None:
            raise FileNotFoundError
        # Ensure the image is binary (0 or 255)
        _, input_image = cv2.threshold(input_image, 127, 255, cv2.THRESH_BINARY)
    except FileNotFoundError:
        print("Image 'FigP0919(UTK).tif' not found. Creating a synthetic image for demonstration.")
        input_image = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(input_image, (50, 50), (150, 150), 255, -1)
        cv2.circle(input_image, (100, 100), 20, 0, -1)
        cv2.line(input_image, (180, 50), (280, 150), 255, 5)

    # Display the original image
    cv2.imshow("Original Image", input_image)

    # Define a 5x5 kernel size, as requested
    ksize_erode = (5, 5)

    print("Applying morphological erosion with different 5x5 kernels...")

    # --- 1. Rectangular Kernel ---
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize_erode)
    eroded_rect = cv2.erode(input_image, kernel_rect, iterations=1)
    cv2.imshow('Eroded (Rectangular Kernel)', eroded_rect)

    # --- 2. Elliptical Kernel ---
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize_erode)
    eroded_ellipse = cv2.erode(input_image, kernel_ellipse, iterations=1)
    cv2.imshow('Eroded (Elliptical Kernel)', eroded_ellipse)

    # --- 3. Cross-shaped Kernel ---
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize_erode)
    eroded_cross = cv2.erode(input_image, kernel_cross, iterations=1)
    cv2.imshow('Eroded (Cross-shaped Kernel)', eroded_cross)

    # --- 4. Diamond-shaped Kernel ---
    kernel_diamond = diamond(radius=2)
    eroded_diamond = cv2.erode(input_image, kernel_diamond, iterations=1)
    cv2.imshow('Eroded (Diamond-shaped Kernel)', eroded_diamond)
    
    # --- Part 2: Erosion comparison with a custom function ---
    print("\nComparing custom and built-in erosion with a 3x3 kernel...")
    # Create a 3x3 square kernel
    structuring_element_3x3 = np.ones((5, 5), dtype=np.uint8) * 255
    
    # Perform the custom morphological fit (erosion)
    eroded_image_custom = morphological_fit(input_image, structuring_element_3x3)
    
    # Perform the built-in OpenCV erosion for comparison
    eroded_image_builtin = cv2.erode(input_image, structuring_element_3x3, iterations=1)
    
    # Display the comparison images
    cv2.imshow("Built-in Eroded Image (3x3)", eroded_image_builtin)
    cv2.imshow("Custom Eroded Image (3x3)", eroded_image_custom)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
