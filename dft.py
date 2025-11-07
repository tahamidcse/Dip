import cv2
import numpy as np
import matplotlib.pyplot as plt
# noisy image fft
def main():
    # Load image in grayscale
    img = cv2.imread('/content/Fig0424(a)(rectangle).tif', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found or path is incorrect.")
        return

    # 1. Compute 2D FFT
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    # 2. Inverse FFT to reconstruct the image
    fft_unshifted = np.fft.ifftshift(fft_shifted)
    reconstructed = np.fft.ifft2(fft_unshifted)
    reconstructed = np.abs(reconstructed).astype(np.uint8)  # Take real part and convert to uint8

    # 3. Compute difference between original and reconstructed
    difference = cv2.absdiff(img, reconstructed)

    # 4. Plot all images
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title("Reconstructed Image (IFFT)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(difference, cmap='hot')
    plt.title("Difference Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
