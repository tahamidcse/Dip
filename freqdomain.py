import cv2
import numpy as np
import matplotlib.pyplot as plt
# noisy high low contrast
def create_circular_mask(shape, radius):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = (X - center_col)**2 + (Y - center_row)**2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[distance <= radius**2] = 1
    return mask

def apply_filter(img, mask):
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    filtered_fft = fft_shifted * mask
    fft_unshifted = np.fft.ifftshift(filtered_fft)
    reconstructed = np.fft.ifft2(fft_unshifted)
    reconstructed = np.abs(reconstructed)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def main():
    # Load image in grayscale
    img = cv2.imread('/content/Fig0333(a)(test_pattern_blurring_orig).tif', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found or path is incorrect.")
        return

    h, w = img.shape

    # Define radius
    radius = 30

    # Create low-pass mask (binary)
    lp_mask = create_circular_mask((h, w), radius)

    # Create high-pass mask as inverse of low-pass
    hp_mask = 255 - lp_mask

    # Apply both filters
    low_pass_img = apply_filter(img, lp_mask)
    high_pass_img = apply_filter(img, hp_mask)

    # Display results
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(low_pass_img, cmap='gray')
    plt.title(f"Low-Pass Filtered (r={radius})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(high_pass_img, cmap='gray')
    plt.title(f"High-Pass Filtered (r={radius})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
