import cv2
import numpy as np
from skimage.morphology import diamond

def erode(input_img, kernel):
    """
    Performs a custom morphological erosion operation with padding.
    """
    input_h, input_w = input_img.shape
    kernel_h, kernel_w = kernel.shape

    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_img = cv2.copyMakeBorder(input_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

    input_normalized = (padded_img > 0).astype(np.uint8)
    kernel_normalized = (kernel > 0).astype(np.uint8)

    output_img = np.zeros_like(input_img, dtype=np.uint8)

    for h in range(input_h):
        for w in range(input_w):
            roi = input_normalized[h:h + kernel_h, w:w + kernel_w]
            if np.all(roi == kernel_normalized):
                output_img[h, w] = 255

    return output_img

def dilate(input_img, kernel):
    """
    Performs a custom morphological dilation operation with padding.
    """
    input_h, input_w = input_img.shape
    kernel_h, kernel_w = kernel.shape
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_img = cv2.copyMakeBorder(input_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

    output_img = np.zeros_like(input_img, dtype=np.uint8)

    input_normalized = (padded_img > 0).astype(np.uint8)
    kernel_normalized = (kernel > 0).astype(np.uint8)

    for h in range(input_h):
        for w in range(input_w):
            roi = input_normalized[h:h + kernel_h, w:w + kernel_w]
            if np.sum(roi * kernel_normalized) > 0:
                output_img[h, w] = 255

    return output_img




print(kernel.astype(int))


if __name__ == '__main__':
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
    # radius = 2 => gives a 5x5 diamond
    kernel_diamond = diamond(radius=2)

    # Perform Black Hat transform for each kernel
    closed_rect = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_rect)
    blackhat_rect = closed_rect - img

    closed_ellipse = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_ellipse)
    blackhat_ellipse = closed_ellipse - img

    closed_cross = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_cross)
    blackhat_cross = closed_cross - img

    # Use the custom functions for the diamond kernel
    closed_diamond = erode(dilate(img, kernel_diamond), kernel_diamond)
    blackhat_diamond = closed_diamond - img
    
    # Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Black Hat (Rectangular)', blackhat_rect)
    cv2.imshow('Black Hat (Elliptical)', blackhat_ellipse)
    cv2.imshow('Black Hat (Cross-shaped)', blackhat_cross)
    cv2.imshow('Black Hat (Diamond-shaped)', blackhat_diamond)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
