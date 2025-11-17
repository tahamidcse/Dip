import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

def calculate_ssim_opencv_skimage(original_path, compressed_path):
    """
    Calculate SSIM between original and compressed images using OpenCV and scikit-image
    """
    # Read images
    original_gray = original_path
    compressed_gray = compressed_path
    
    # Check if images are loaded properly
  
    # Convert to grayscale for SSIM calculation
    
    
    # Calculate SSIM
    ssim_index, ssim_map = ssim(original_gray, compressed_gray,multichannel=True, full=True)
    
    return ssim_index



class GrayLevelQuantizer:
    def decrease_resolution(self, image, number_of_bits):
        step = 255 / (2**number_of_bits - 1)
        height, width = image.shape
        decreased_image = image.copy()

        for r in range(height):
            for c in range(width):
                decreased_image[r, c] = round(image[r, c] / step) * step
        
        return decreased_image
    
    def demonstrate_gray_quantization(self, image):
        plt.figure(figsize=(13, 8))
        for k in range(1, 9):
            plt.subplot(2, 4, k)
            number_of_bits = 9 - k
            decreased_image = self.decrease_resolution(image, number_of_bits)
            plt.imshow(decreased_image, cmap='gray')
            plt.title(f"{number_of_bits}-Bits Image")
        plt.show()


class RGB332Compressor(GrayLevelQuantizer):

    def compress_332_rgb(self, image):
        b, g, r = cv2.split(image)
        r_q = self.decrease_resolution(r, 3)
        g_q = self.decrease_resolution(g, 3)
        b_q = self.decrease_resolution(b, 2)
        return cv2.merge([b_q, g_q, r_q])

    def get_quantization_levels(self, number_of_bits):
        step = 255 / (2**number_of_bits - 1)
        return [round(i * step) for i in range(2**number_of_bits)]

    def calculate_compression_stats(self, original_img, compressed_img):
        
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        compressed_rgb = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB)
        
        original_size = original_img.nbytes
        compressed_size = compressed_img.nbytes

        original_bpp = 24
        compressed_bpp = 8

        compression_ratio = original_bpp / compressed_bpp
        actual_ratio = original_size / compressed_size

        mse = np.mean((original_img.astype(float) - compressed_img.astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        ssimrgb=calculate_ssim_opencv_skimage(original_rgb,compressed_rgb)
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'original_bpp': original_bpp,
            'compressed_bpp': compressed_bpp,
            'theoretical_ratio': compression_ratio,
            'actual_ratio': actual_ratio,
            'mse': mse,
            'psnr': psnr,
            'ssimrgb':ssimrgb
            
        }


    @staticmethod
    def display_compression_process(original, compressed, stats):
        
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        #
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 4, 1)
        plt.imshow(original_rgb)
        plt.title(f'Original\n24-bit RGB\n{stats["original_size"]} bytes')
        plt.axis('off')

        plt.subplot(2, 4, 6)
        plt.imshow(compressed_rgb)
        plt.title(f'Compressed\n3-3-2 bit RGB\n{stats["compressed_size"]} bytes')
        plt.axis('off')

        plt.subplot(2, 4, 7)
        diff = cv2.absdiff(original, compressed)
        diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        plt.imshow(diff_rgb)
        plt.title(f'Difference\nMSE: {stats["mse"]:.2f}')
        plt.axis('off')

        plt.subplot(2, 4, 8)
        plt.axis('off')
        txt = (
            f"Compression Statistics:\n\n"
            f"Original: {stats['original_bpp']} bpp\n"
            f"Compressed: {stats['compressed_bpp']} bpp\n"
            f"Ratio: {stats['theoretical_ratio']:.1f}:1\n"
            f"MSE: {stats['mse']:.2f}\n"
            f"PSNR: {stats['psnr']:.2f} dB\n"
            f"SSIM ={stats['ssimrgb']:.2f}"
        )
        plt.text(0.1, 0.5, txt, fontsize=12)

        plt.tight_layout()
        plt.show()


def main():

    img_path = 'rgb_palette_24bit.png'
    original_img = cv2.imread(img_path)

    if original_img is None:
        print("Image not found, generating sample image...")
        original_img = np.zeros((300, 400, 3), dtype=np.uint8)
        for i in range(300):
            for j in range(400):
                original_img[i, j, 0] = int(255 * j / 400)
                original_img[i, j, 1] = int(255 * i / 300)
                original_img[i, j, 2] = int(255 * (i+j) / 700)

    compressor = RGB332Compressor()

    #compressor.demonstrate_gray_quantization(gray_img)

    #Compress image FIRST
    compressed_img = compressor.compress_332_rgb(original_img)

    #  Now compute stats
    stats = compressor.calculate_compression_stats(original_img, compressed_img)

    #  Display output
    compressor.display_compression_process(original_img, compressed_img, stats)
    
    
    
    print("\n=== COMPRESSION SUMMARY ===")
    print(f"PSNR = {stats['psnr']:.2f} dB")
    print(f"MSE  = {stats['mse']:.2f}")
    print(f"Compression Ratio = {stats['theoretical_ratio']:.1f}:1")
    print(f"SSIM ={stats['ssimrgb']:.2f}")

if __name__ == '__main__':
    main()

    
