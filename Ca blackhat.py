import cv2
import numpy as np
from skimage.morphology import diamond

def erode(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    # Pad with white for erosion (255)
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)

    output = np.zeros_like(img, dtype=np.uint8)
    kernel_mask = (kernel > 0)

    for y in range(h):
        for x in range(w):
            roi = padded[y:y + kh, x:x + kw]
            output[y, x] = np.min(roi[kernel_mask])
    return output

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


# Load an image (e.g., a binary or grayscale image)
img = cv2.imread('testing1.jpg', cv2.IMREAD_GRAYSCALE)

_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Structuring elements
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15)) 
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) 
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kernel_diamond = diamond(radius=2).astype(np.uint8)

# --- Cross Kernel ---
blackhat_cv_cross = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_cross)
closed_cross = erode(dilate(img, kernel_cross), kernel_cross)
blackhat_custom_cross = closed_cross - img

# --- Rect Kernel ---
blackhat_cv_rect = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_rect)
closed_rect = erode(dilate(img, kernel_rect), kernel_rect)
blackhat_custom_rect = closed_rect - img

# --- Ellipse Kernel ---
blackhat_cv_ellipse = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_ellipse)
closed_ellipse = erode(dilate(img, kernel_ellipse), kernel_ellipse)
blackhat_custom_ellipse = closed_ellipse - img

# --- Diamond Kernel ---
blackhat_cv_diamond = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_diamond)
closed_diamond = erode(dilate(img, kernel_diamond), kernel_diamond)
blackhat_custom_diamond = closed_diamond - img

cv2.imshow("Original Image", img)

cv2.imshow("BlackHat OpenCV - Cross", blackhat_cv_cross)
cv2.imshow("BlackHat Custom - Cross", blackhat_custom_cross)

cv2.imshow("BlackHat OpenCV - Rect", blackhat_cv_rect)
cv2.imshow("BlackHat Custom - Rect", blackhat_custom_rect)

cv2.imshow("BlackHat OpenCV - Ellipse", blackhat_cv_ellipse)
cv2.imshow("BlackHat Custom - Ellipse", blackhat_custom_ellipse)

cv2.imshow("BlackHat OpenCV - Diamond", blackhat_cv_diamond)
cv2.imshow("BlackHat Custom - Diamond", blackhat_custom_diamond)



cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows() 

cv2.destroyAllWindows()
