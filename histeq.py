import cv2
import matplotlib.pyplot as plt
import numpy as np

# Custom rounding function (although not necessary — numpy's round works fine)
def custom_round(x):
    y = int(x)
    if x <= (y + 0.5):
        return y
    else:
        return y + 1

# Histogram Equalization Function
def equalize_histogram(img, color_channel=None):  # color_channel not used, kept for compatibility
    L = 256
    pixel_count = np.zeros((L,), dtype=np.uint)

    h, w = img.shape

    # Count the number of pixels for each intensity level
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            pixel_count[pixel_value] += 1

    # Normalize the histogram (PDF)
    total_pixels = h * w
    pdf = pixel_count / total_pixels

    # Compute the CDF
    cdf = pdf.cumsum()

    # Map the CDF values to [0, 255]
    csum = [custom_round(i * (L - 1)) for i in cdf]

    # Create new equalized image
    img_equalized = img.copy()
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            img_equalized[i, j] = csum[pixel_value]

    return img_equalized

# Main function
def main():
    # Load grayscale images
    img1 = cv2.imread('Fig0316(1).tif', 0)
    img2 = cv2.imread('Fig0316(2).tif', 0)
    img3 = cv2.imread('Fig0316(3).tif', 0)
    img4 = cv2.imread('Fig0316(4).tif', 0)

    # Apply OpenCV histogram equalization
    opencv_hist1 = cv2.equalizeHist(img1)

    # Apply custom histogram equalization
    custom_hist1 = equalize_histogram(img1, 'gray')

    # Display results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("OpenCV Equalized")
    plt.imshow(opencv_hist1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Custom Equalized")
    plt.imshow(custom_hist1, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


def display_imgset(img_set, color_set, title_set='', row=1, col=1):
    plt.figure(figsize=(8, 4))
    k = 1
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if k > len(img_set):
                break
            plt.subplot(row, col, k)
            img = img_set[k - 1]
            if len(img.shape) == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap=color_set[k - 1])
            if title_set[k - 1] != '':
                plt.title(title_set[k - 1])
            plt.axis('off')
            k += 1
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
   
