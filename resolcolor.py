import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_grayscale_resolution(img, bits):
    """
    Reduce grayscale resolution using stepsize method
    stepsize = 255 / (2^bits - 1)
    r = (r / stepsize) * stepsize
    """
    stepsize = 255 / (2**bits - 1)
    reduced = (img / stepsize).astype(np.uint8) * stepsize
    return reduced.astype(np.uint8)

def reduce_rgb_resolution(img, r_bits, g_bits, b_bits):
    """
    Reduce RGB image resolution by reducing bit depth for each channel
    using stepsize method
    """
    # Split channels
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # Calculate stepsize for each channel
    r_stepsize = 255 / (2**r_bits - 1)
    g_stepsize = 255 / (2**g_bits - 1)
    b_stepsize = 255 / (2**b_bits - 1)
    
    # Reduce resolution for each channel
    r_reduced = (r / r_stepsize).astype(np.uint8) * r_stepsize
    g_reduced = (g / g_stepsize).astype(np.uint8) * g_stepsize
    b_reduced = (b / b_stepsize).astype(np.uint8) * b_stepsize
    
    # Combine channels
    reduced_img = np.stack([r_reduced, g_reduced, b_reduced], axis=2)
    return reduced_img.astype(np.uint8)

def reduce_rgb_uniform_resolution(img, bits):
    """
    Reduce RGB image resolution with uniform bit depth for all channels
    """
    return reduce_rgb_resolution(img, bits, bits, bits)

def calculate_compression_stats(original_img, reduced_img, bits_per_channel):
    """Calculate compression statistics"""
    original_size = original_img.shape[0] * original_img.shape[1] * 24  # 24 bits per pixel
    if len(bits_per_channel) == 1:
        reduced_size = original_img.shape[0] * original_img.shape[1] * bits_per_channel[0] * 3
    else:
        reduced_size = original_img.shape[0] * original_img.shape[1] * sum(bits_per_channel)
    
    compression_ratio = original_size / reduced_size
    compression_percentage = (1 - reduced_size / original_size) * 100
    
    return compression_ratio, compression_percentage, reduced_size

def plot_resolution_reduction_comparison(img):
    """Plot comparison of different resolution reductions"""
    
    # Different bit configurations to test
    configurations = [
        ('24-bit Original', (8, 8, 8)),
        ('8-bit/ch (Uniform)', (3, 3, 3)),
        ('8-bit Total (3-3-2)', (3, 3, 2)),
        ('6-bit Total (2-2-2)', (2, 2, 2)),
        ('4-bit Total (2-1-1)', (2, 1, 1))
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (title, bits_config) in enumerate(configurations):
        if idx >= len(axes):
            break
            
        if title == '24-bit Original':
            # Original image
            processed_img = img.copy()
            bits_used = 24
        else:
            # Apply resolution reduction
            if len(bits_config) == 3:
                processed_img = reduce_rgb_resolution(img, bits_config[0], bits_config[1], bits_config[2])
                bits_used = sum(bits_config)
            else:
                processed_img = reduce_rgb_uniform_resolution(img, bits_config[0])
                bits_used = bits_config[0] * 3
        
        # Calculate compression stats
        comp_ratio, comp_pct, reduced_size = calculate_compression_stats(img, processed_img, bits_config)
        
        # Display image
        axes[idx].imshow(processed_img)
        axes[idx].set_title(f'{title}\n{bits_used} bits/pixel\nCompression: {comp_pct:.1f}%')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(configurations), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def demonstrate_stepsize_calculation():
    """Demonstrate stepsize calculation for different bit depths"""
    print("STEP SIZE CALCULATIONS FOR DIFFERENT BIT DEPTHS")
    print("=" * 50)
    print(f"{'Bits':<6} {'Levels':<8} {'Step Size':<12} {'Possible Values'}")
    print("-" * 50)
    
    for bits in [1, 2, 3, 4, 5, 6, 7, 8]:
        levels = 2**bits
        stepsize = 255 / (levels - 1)
        values = [i * stepsize for i in range(levels)]
        print(f"{bits:<6} {levels:<8} {stepsize:<12.2f} {[int(v) for v in values[:4]]}...")

def show_channel_histograms(original_img, reduced_img, title):
    """Show histograms of original vs reduced images"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']
    
    for i in range(3):
        # Original histogram
        axes[i, 0].hist(original_img[:,:,i].ravel(), bins=256, color=colors[i], alpha=0.7)
        axes[i, 0].set_title(f'Original {channel_names[i]} Channel')
        axes[i, 0].set_xlim(0, 255)
        
        # Reduced histogram
        axes[i, 1].hist(reduced_img[:,:,i].ravel(), bins=256, color=colors[i], alpha=0.7)
        axes[i, 1].set_title(f'Reduced {channel_names[i]} Channel - {title}')
        axes[i, 1].set_xlim(0, 255)
    
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    # Load or create image
    try:
        img = cv2.imread('1000100628.jpg')
        if img is None:
            raise FileNotFoundError
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print("Creating sample image...")
        img = create_sample_image()
    
    print("RGB IMAGE RESOLUTION REDUCTION USING STEPSIZE METHOD")
    print("=" * 60)
    
    # Demonstrate stepsize calculations
    demonstrate_stepsize_calculation()
    
    # Show original image info
    print(f"\nOriginal Image: {img.shape[1]} x {img.shape[0]} pixels")
    print(f"Original Color Depth: 24 bits/pixel (8-8-8)")
    print(f"Possible Colors: 16,777,216")
    
    # Test different reductions
    print("\nTESTING DIFFERENT RESOLUTION REDUCTIONS:")
    print("-" * 50)
    
    # 8-bit uniform (3-3-3)
    img_8bit_uniform = reduce_rgb_uniform_resolution(img, 3)
    comp_ratio, comp_pct, size = calculate_compression_stats(img, img_8bit_uniform, (3,))
    print(f"8-bit uniform (3-3-3): {comp_pct:.1f}% compression")
    print(f"  Levels per channel: {2**3} = 8")
    print(f"  Total colors: 8 × 8 × 8 = 512")
    
    # 8-bit custom (3-3-2)
    img_8bit_custom = reduce_rgb_resolution(img, 3, 3, 2)
    comp_ratio, comp_pct, size = calculate_compression_stats(img, img_8bit_custom, (3, 3, 2))
    print(f"8-bit custom (3-3-2): {comp_pct:.1f}% compression")
    print(f"  Levels: R={2**3}=8, G={2**3}=8, B={2**2}=4")
    print(f"  Total colors: 8 × 8 × 4 = 256")
    
    # 4-bit custom (2-1-1)
    img_4bit_custom = reduce_rgb_resolution(img, 2, 1, 1)
    comp_ratio, comp_pct, size = calculate_compression_stats(img, img_4bit_custom, (2, 1, 1))
    print(f"4-bit custom (2-1-1): {comp_pct:.1f}% compression")
    print(f"  Levels: R={2**2}=4, G={2**1}=2, B={2**1}=2")
    print(f"  Total colors: 4 × 2 × 2 = 16")
    
    # Create visual comparison
    plot_resolution_reduction_comparison(img)
    
    # Show histogram comparison for 8-bit custom
    show_channel_histograms(img, img_8bit_custom, "8-bit (3-3-2)")

def create_sample_image():
    """Create a sample image with various colors"""
    height, width = 300, 400
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create color gradients
    for i in range(height):
        for j in range(width):
            img[i, j, 0] = int((i / height) * 255)        # Red vertical gradient
            img[i, j, 1] = int((j / width) * 255)         # Green horizontal gradient
            img[i, j, 2] = (i + j) % 256                  # Blue diagonal pattern
    
    # Add solid color patches
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    for idx, color in enumerate(colors):
        x, y = 50 + (idx % 3) * 100, 50 + (idx // 3) * 80
        cv2.rectangle(img, (x, y), (x+60, y+40), color, -1)
    
    return img

if __name__ == "__main__":
    main()
