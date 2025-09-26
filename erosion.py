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
# Load image
img = cv2.imread('Fig0905(a)(wirebond-mask) .tif', cv2.IMREAD_GRAYSCALE)

# Ensure binary image
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Structuring elements
ksize = (15, 15)
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
kernel_diamond = diamond(radius=2)

# === OpenCV Erosion ===
eroded_rect_cv = cv2.erode(img, kernel_rect)
eroded_ellipse_cv = cv2.erode(img, kernel_ellipse)
eroded_cross_cv = cv2.erode(img, kernel_cross)
eroded_diamond_cv = cv2.erode(img, kernel_diamond)

# === Custom Erosion ===
eroded_rect_custom = erode(img, kernel_rect)
eroded_ellipse_custom = erode(img, kernel_ellipse)
eroded_cross_custom = erode(img, kernel_cross)
eroded_diamond_custom = erode(img, kernel_diamond)

# === Display ===
cv2.imshow("Original", img)

cv2.imshow("OpenCV Erode - Rect", eroded_rect_cv)
cv2.imshow("Custom Erode - Rect", eroded_rect_custom)

cv2.imshow("OpenCV Erode - Ellipse", eroded_ellipse_cv)
cv2.imshow("Custom Erode - Ellipse", eroded_ellipse_custom)

cv2.imshow("OpenCV Erode - Cross", eroded_cross_cv)
cv2.imshow("Custom Erode - Cross", eroded_cross_custom)

cv2.imshow("OpenCV Erode - Diamond", eroded_diamond_cv)
cv2.imshow("Custom Erode - Diamond", eroded_diamond_custom)



cv2.waitKey(0)
cv2.destroyAllWindows()
