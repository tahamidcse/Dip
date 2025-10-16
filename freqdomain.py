import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_contrast(image, contrast_level):
    """Adjust image contrast"""
    if contrast_level == 'low':
        alpha = 0.01  # Lower contrast
    elif contrast_level == 'high':
        alpha = 1.99  # Higher contrast
    else:  # normal
        alpha = 1.0  # Original contrast
    
    beta = 0  # Brightness adjustment
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def create_circular_mask(shape, radius):
    rows, cols = shape
    center = (cols // 2, rows // 2)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)  # -1 means filled circle
    return mask

def create_band_pass_mask(shape, inner_radius, outer_radius):
    rows, cols = shape
    center = (cols // 2, rows // 2)
    
    # Create outer circle
    outer_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(outer_mask, center, outer_radius, 1, -1)
    
    # Create inner circle (to be subtracted)
    inner_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(inner_mask, center, inner_radius, 1, -1)
    
    # Band-pass = outer circle minus inner circle
    band_mask = outer_mask - inner_mask
    return band_mask

def freq_domain_filter(img):
    fft = np.fft.fft2(img.astype(float))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)
    return log_magnitude

def apply_filter(img, mask):
    fft = np.fft.fft2(img.astype(float))
    fft_shifted = np.fft.fftshift(fft)
    filtered_fft = fft_shifted * mask
    fft_unshifted = np.fft.ifftshift(filtered_fft)
    reconstructed = np.fft.ifft2(fft_unshifted)
    reconstructed = np.abs(reconstructed)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def normalize_for_display(data):
    """Normalize data for proper display"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def main():
    # Load image in grayscale
    img = cv2.imread('/content/Fig0431(d)(blown_ic_crop).tif', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found or path is incorrect.")
        return

    print(f"Original image shape: {img.shape}")

    # Create contrast versions
    low_contrast_img = adjust_contrast(img, 'low')
    high_contrast_img = adjust_contrast(img, 'high')
    normal_contrast_img = adjust_contrast(img, 'normal')  # This should be same as original

    # Compute FFT magnitude spectra
    low_contrast_magnitude = freq_domain_filter(low_contrast_img)
    high_contrast_magnitude = freq_domain_filter(high_contrast_img)
    normal_contrast_magnitude = freq_domain_filter(img)  # Fixed variable name

    # Save normalized FFT images for visualization
    cv2.imwrite('low_contrast_fft.jpg', normalize_for_display(low_contrast_magnitude) * 255)
    cv2.imwrite('high_contrast_fft.jpg', normalize_for_display(high_contrast_magnitude) * 255)
    cv2.imwrite('normal_contrast_fft.jpg', normalize_for_display(normal_contrast_magnitude) * 255)

    h, w = img.shape

    # Define radii
    low_radius = 30
    high_radius = 60

    # Create masks
    lp_mask = create_circular_mask((h, w), low_radius)
    hp_mask = 1 - lp_mask  # Inverse (binary)
    bp_mask = create_band_pass_mask((h, w), low_radius, high_radius)

    # Apply filters to original image
    low_pass_img = apply_filter(img, lp_mask)
    high_pass_img = apply_filter(img, hp_mask)
    band_pass_img = apply_filter(img, bp_mask)

    # Apply filters to contrast variants
    low_pass_img_low_contrast = apply_filter(low_contrast_img, lp_mask)
    high_pass_img_low_contrast = apply_filter(low_contrast_img, hp_mask)
    band_pass_img_low_contrast = apply_filter(low_contrast_img, bp_mask)
    
    low_pass_img_high_contrast = apply_filter(high_contrast_img, lp_mask)
    high_pass_img_high_contrast = apply_filter(high_contrast_img, hp_mask)
    band_pass_img_high_contrast = apply_filter(high_contrast_img, bp_mask)
    
    low_pass_img_normal_contrast = apply_filter(normal_contrast_img, lp_mask)
    high_pass_img_normal_contrast = apply_filter(normal_contrast_img, hp_mask)
    band_pass_img_normal_contrast = apply_filter(normal_contrast_img, bp_mask)

    # Save filtered images
    cv2.imwrite('low_pass_img_lowc.jpg', low_pass_img_low_contrast)
    cv2.imwrite('high_pass_img_lowc.jpg', high_pass_img_low_contrast)
    cv2.imwrite('band_pass_img_lowc.jpg', band_pass_img_low_contrast)
    cv2.imwrite('low_pass_img_highc.jpg', low_pass_img_high_contrast)
    cv2.imwrite('high_pass_img_highc.jpg', high_pass_img_high_contrast)
    cv2.imwrite('band_pass_img_highc.jpg', band_pass_img_high_contrast)
    cv2.imwrite('low_pass_img_normalc.jpg', low_pass_img_normal_contrast)
    cv2.imwrite('high_pass_img_normalc.jpg', high_pass_img_normal_contrast)
    cv2.imwrite('band_pass_img_normalc.jpg', band_pass_img_normal_contrast)

    # Display results - Contrast comparison
    plt.figure(figsize=(20, 12))

    # Row 1: Original images with different contrasts
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(low_contrast_img, cmap='gray')
    plt.title("Low Contrast")
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.imshow(high_contrast_img, cmap='gray')
    plt.title("High Contrast")
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.imshow(normal_contrast_img, cmap='gray')
    plt.title("Normal Contrast")
    plt.axis('off')

    # Row 2: FFT magnitude spectra
    plt.subplot(3, 4, 5)
    plt.imshow(normal_contrast_magnitude, cmap='viridis')
    plt.title("FFT - Original")
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(low_contrast_magnitude, cmap='viridis')
    plt.title("FFT - Low Contrast")
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(high_contrast_magnitude, cmap='viridis')
    plt.title("FFT - High Contrast")
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.imshow(lp_mask, cmap='gray')
    plt.title("Low Pass Mask")
    plt.axis('off')

    # Row 3: Filtered results example (using normal contrast)
    plt.subplot(3, 4, 9)
    plt.imshow(low_pass_img_normal_contrast, cmap='gray')
    plt.title("Low Pass Filtered")
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(high_pass_img_normal_contrast, cmap='gray')
    plt.title("High Pass Filtered")
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.imshow(band_pass_img_normal_contrast, cmap='gray')
    plt.title("Band Pass Filtered")
    plt.axis('off')

    plt.subplot(3, 4, 12)
    plt.imshow(bp_mask, cmap='gray')
    plt.title("Band Pass Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("All images processed and saved successfully!")

if __name__ == "__main__":
    main()
