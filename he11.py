import cv2
import numpy as np
import matplotlib.pyplot as plt

def prepare_histogram(img, color_channel='gray'):
    """Plots the histogram and CDF of a grayscale image."""
    L = 256
    pixel_count = np.zeros((L,), dtype=np.uint)

    h, w = img.shape
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            pixel_count[pixel_value] += 1

    pdf = pixel_count / (h * w)
    cdf = pdf.cumsum()
    x = np.arange(L)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, pixel_count, 'r')
    plt.title('Histogram of ' + color_channel + ' Channel')
    plt.xlabel('Pixel Value')
    plt.ylabel('Pixel Count')

    plt.subplot(1, 2, 2)
    plt.plot(x, cdf, 'b')
    plt.title('CDF of ' + color_channel + ' Channel')
    plt.xlabel('Pixel Value')
    plt.ylabel('Cumulative Probability')

    plt.tight_layout()
    plt.show()
    plt.close()


def equalize_histogram(img, color_channel='gray'):
    """Performs histogram equalization manually and plots the result."""
    L = 256
    pixel_count = np.zeros((L,), dtype=np.uint)

    h, w = img.shape
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            pixel_count[pixel_value] += 1

    total_pixels = h * w
    pdf = pixel_count / total_pixels

    # Calculate cumulative distribution function (CDF)
    cdf = np.cumsum(pdf)
    mapping = np.round(cdf * (L - 1)).astype(np.uint8)

    # Apply the equalization mapping
    equalized_img = mapping[img]

    # New histogram of equalized image
    equalized_pixel_count = np.zeros((L,), dtype=np.uint)
    for i in range(h):
        for j in range(w):
            equalized_pixel_count[equalized_img[i, j]] += 1

    pdfe = equalized_pixel_count / total_pixels
    cdfe = np.cumsum(pdfe)
    x = np.arange(L)

    # Plotting
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(x, pixel_count, 'r')
    plt.title('Original Histogram')

    plt.subplot(2, 2, 2)
    plt.plot(x, equalized_pixel_count, 'g')
    plt.title('Equalized Histogram (Manual)')

    plt.subplot(2, 2, 3)
    plt.plot(x, cdfe, 'b')
    plt.title('Equalized CDF (Manual)')

    plt.subplot(2, 2, 4)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('Equalized Image (Manual)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

    return equalized_img  # return for further comparison


def compare_with_cv2_equalization(img):
    """Compares manual equalization with OpenCV's built-in function."""
    equalized_cv2 = cv2.equalizeHist(img)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(equalized_cv2, cmap='gray')
    plt.title('OpenCV Equalized Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    cv2_hist = cv2.calcHist([equalized_cv2], [0], None, [256], [0, 256])
    plt.plot(cv2_hist, color='purple')
    plt.title('Histogram (OpenCV Equalized)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()
    plt.close()


# -------------------
# Example usage
# -------------------

# Load a grayscale image
img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your image path

# Plot original histogram
prepare_histogram(img, 'gray')

# Manual Histogram Equalization
equalized_manual = equalize_histogram(img, 'gray')

# Compare with OpenCV
compare_with_cv2_equalization(img)
