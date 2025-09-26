import numpy as np
import cv2
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


if __name__ == '__main__':
    # Load an image

    img = cv2.imread('testing2.png', cv2.IMREAD_GRAYSCALE)
    
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Define a structuring element
    #kernel = np.ones((15, 15), np.uint8)

    # Perform morphological opening with built-in functions
    ksize = (15, 15)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    kernel_diamond = diamond(radius=2)
    
    # Perform morphological opening with custom functions
   
    
    # Perform Top-Hat transform
    tophat_rect = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_rect)
    tophat_ellipse = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_ellipse)
    tophat_cross = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_cross)
    tophat_diamond = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_diamond)
    
    eroded_image_customr = erode(img, kernel_rect)
    opened_image_customr = dilate(eroded_image_customr, kernel_rect)
    eroded_image_custome = erode(img, kernel_ellipse)
    opened_image_custome = dilate(eroded_image_custome, kernel_ellipse)
    eroded_image_customc = erode(img, kernel_cross)
    opened_image_customc = dilate(eroded_image_customc, kernel_cross)
    eroded_image_customd = erode(img, kernel_diamond)
    opened_image_customd = dilate(eroded_image_customd, kernel_diamond)
    
   
    tophat_rect2 = img - opened_image_customr
    tophat_ellipse2 = img - opened_image_custome
    tophat_cross2 = img - opened_image_customc
    tophat_diamond2 = img - opened_image_customd

    
    # Display the images
    cv2.imshow('Original Image', img)
    cv2.imshow('Top-Hat (rect Custom)', tophat_rect2)
    cv2.imshow('Top-Hat (rect CV2)', tophat_rect)
    cv2.imshow('Top-Hat (ellipse)custom', tophat_ellipse2)
   
    cv2.imshow('Top-HatCV2 (elliplse)', tophat_ellipse)
    cv2.imshow('Top-Hat (cross)custom', tophat_cross2)
   
    cv2.imshow('Top-HatCV2 (cross)', tophat_cross)
    cv2.imshow('Top-Hat (diamond)custom', tophat_diamond2)
   
    cv2.imshow('Top-HatCV2 (diamond)', tophat_diamond)
    
    
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
