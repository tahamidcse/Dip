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

# Load or create image
img = cv2.imread('morph.png', cv2.IMREAD_GRAYSCALE)


_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

ksize = (15, 15)
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
kernel_diamond = diamond(radius=2)

# === OpenCV Openings ===
opened_rect_cv = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_rect)
opened_ellipse_cv = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_ellipse)
opened_cross_cv = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_cross)
opened_diamond_cv = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_diamond)

# === Custom Openings ===
opened_rect_custom = dilate(erode(img, kernel_rect), kernel_rect)
opened_ellipse_custom = dilate(erode(img, kernel_ellipse), kernel_ellipse)
opened_cross_custom = dilate(erode(img, kernel_cross), kernel_cross)
opened_diamond_custom = dilate(erode(img, kernel_diamond), kernel_diamond)

# === Display ===
cv2.imshow("Original", img)

cv2.imshow("OpenCV Open - Rect", opened_rect_cv)
cv2.imshow("Custom Open - Rect", opened_rect_custom)

cv2.imshow("OpenCV Open - Ellipse", opened_ellipse_cv)
cv2.imshow("Custom Open - Ellipse", opened_ellipse_custom)

cv2.imshow("OpenCV Open - Cross", opened_cross_cv)
cv2.imshow("Custom Open - Cross", opened_cross_custom)

cv2.imshow("OpenCV Open - Diamond", opened_diamond_cv)
cv2.imshow("Custom Open - Diamond", opened_diamond_custom)



cv2.waitKey(0)
cv2.destroyAllWindows()
