import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Mask Generators
# -----------------------------
def adjust_contrast(image, contrast_level):
    """Adjust image contrast"""
    if contrast_level == 'low':
        alpha = 0.5  # Lower contrast
    elif contrast_level == 'high':
        alpha = 2.0  # Higher contrast
    else:  # normal
        alpha = 1.0  # Original contrast
    
    beta = 0  # Brightness adjustment
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def create_smooth_circular_mask(shape):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u - center_row, v - center_col, indexing='ij')
    distance = np.sqrt(U**2 + V**2)
    return distance

def butter_mask(shape, cutoff=40, n=2):
    D = create_smooth_circular_mask(shape)
    return 1 / (1 + (D / cutoff)**(2 * n))

def gaussian_mask(shape, cutoff=40):
    D = create_smooth_circular_mask(shape)
    return np.exp(-(D**2) / (2 * (cutoff**2)))

def butter_band_pass_mask(shape, inner_radius=20, outer_radius=60, n=2):
    D = create_smooth_circular_mask(shape)
    D0 = (inner_radius + outer_radius) / 2
    W = outer_radius - inner_radius
    mask = 1 / (1 + ((D**2 - D0**2) / (D * W + 1e-5))**(2 * n))
    return mask

def gauss_band_pass_mask(shape, inner_radius=20, outer_radius=60):
    D = create_smooth_circular_mask(shape)
    D0 = (inner_radius + outer_radius) / 2
    W = outer_radius - inner_radius
    mask = np.exp(-((D**2 - D0**2)**2) / (2 * (D0 * W)**2 + 1e-5))
    return mask

# -----------------------------
# Filtering Function
# -----------------------------
def apply_filter(img, mask):
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    filtered_fft = fft_shifted * mask
    fft_unshifted = np.fft.ifftshift(filtered_fft)
    reconstructed = np.fft.ifft2(fft_unshifted)
    reconstructed = np.abs(reconstructed)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

# -----------------------------
# Main Script
# -----------------------------
def main():
    # Load grayscale image
    img = cv2.imread('/content/Fig0441(a)(characters_test_pattern).tif', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found or path is incorrect.")
        return

    shape = img.shape

    # Create contrast variants
    contrast_levels = ['low', 'normal', 'high']
    contrast_imgs = {level: adjust_contrast(img, level) for level in contrast_levels}

    butter_ns = [1, 2, 5]  # Test Butterworth filters with different n values

    for level in contrast_levels:
        current_img = contrast_imgs[level]
        print(f"\nProcessing {level}-contrast image...")

        plt.figure(figsize=(18, 12))
        plt.suptitle(f'{level.capitalize()} Contrast Image Filters', fontsize=16)

        # Gaussian Filters
        gauss_lpf = gaussian_mask(shape, cutoff=40)
        gauss_hpf = 1 - gauss_lpf
        gauss_bpf = gauss_band_pass_mask(shape, inner_radius=20, outer_radius=60)

        filtered_gauss_lpf = apply_filter(current_img, gauss_lpf)
        filtered_gauss_hpf = apply_filter(current_img, gauss_hpf)
        filtered_gauss_bpf = apply_filter(current_img, gauss_bpf)

        # Plot Gaussian filters
        plt.subplot(3, 4, 1)
        plt.imshow(current_img, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(3, 4, 2)
        plt.imshow(filtered_gauss_lpf, cmap='gray')
        plt.title('Gaussian LPF')
        plt.axis('off')

        plt.subplot(3, 4, 3)
        plt.imshow(filtered_gauss_hpf, cmap='gray')
        plt.title('Gaussian HPF')
        plt.axis('off')

        plt.subplot(3, 4, 4)
        plt.imshow(filtered_gauss_bpf, cmap='gray')
        plt.title('Gaussian BPF')
        plt.axis('off')

        # Butterworth filters for different n
        for idx, n in enumerate(butter_ns):
            butter_lpf = butter_mask(shape, cutoff=40, n=n)
            butter_hpf = 1 - butter_lpf
            butter_bpf = butter_band_pass_mask(shape, inner_radius=20, outer_radius=60, n=n)

            filtered_butter_lpf = apply_filter(current_img, butter_lpf)
            filtered_butter_hpf = apply_filter(current_img, butter_hpf)
            filtered_butter_bpf = apply_filter(current_img, butter_bpf)

            # Plot Butterworth filters
            row = idx + 2
            plt.subplot(3, 4, row * 4 - 3)
            plt.imshow(filtered_butter_lpf, cmap='gray')
            plt.title(f'Butterworth LPF (n={n})')
            plt.axis('off')

            plt.subplot(3, 4, row * 4 - 2)
            plt.imshow(filtered_butter_hpf, cmap='gray')
            plt.title(f'Butterworth HPF (n={n})')
            plt.axis('off')

            plt.subplot(3, 4, row * 4 - 1)
            plt.imshow(filtered_butter_bpf, cmap='gray')
            plt.title(f'Butterworth BPF (n={n})')
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# Run the script
if __name__ == '__main__':
    main()
