import cv2
import numpy as np
from skimage.morphology import diamond

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

# Load image
img = cv2.imread('Fig0907(a)(text_gaps_1_and_2_pixels).tif', cv2.IMREAD_GRAYSCALE)

# Ensure binary image
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Structuring elements
ksize = (5, 5)
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
kernel_diamond = diamond(radius=2).astype(np.uint8)

# === OpenCV Dilation ===
dilated_rect_cv = cv2.dilate(img, kernel_rect)
dilated_ellipse_cv = cv2.dilate(img, kernel_ellipse)
dilated_cross_cv = cv2.dilate(img, kernel_cross)
dilated_diamond_cv = cv2.dilate(img, kernel_diamond)

# === Custom Dilation ===
dilated_rect_custom = dilate(img, kernel_rect)
dilated_ellipse_custom = dilate(img, kernel_ellipse)
dilated_cross_custom = dilate(img, kernel_cross)
dilated_diamond_custom = dilate(img, kernel_diamond)

# === Display ===
cv2.imshow("Original", img)

cv2.imshow("OpenCV Dilate - Rect", dilated_rect_cv)
cv2.imshow("Custom Dilate - Rect", dilated_rect_custom)

cv2.imshow("OpenCV Dilate - Ellipse", dilated_ellipse_cv)
cv2.imshow("Custom Dilate - Ellipse", dilated_ellipse_custom)

cv2.imshow("OpenCV Dilate - Cross", dilated_cross_cv)
cv2.imshow("Custom Dilate - Cross", dilated_cross_custom)

cv2.imshow("OpenCV Dilate - Diamond", dilated_diamond_cv)
cv2.imshow("Custom Dilate - Diamond", dilated_diamond_custom)



cv2.waitKey(0)
cv2.destroyAllWindows()
