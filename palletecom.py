import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import log2, ceil, floor

class DynamicPaletteColorReducer:
    """Class for dynamic palette-based color depth reduction"""
    
    def __init__(self, total_bits=8):
        self.total_bits = total_bits
        self.r_bits, self.g_bits, self.b_bits = self.calculate_bit_allocation(total_bits)
        
        # Define color levels for each channel
        self.levels_r = self.calculate_quantization_levels(self.r_bits)
        self.levels_g = self.calculate_quantization_levels(self.g_bits)
        self.levels_b = self.calculate_quantization_levels(self.b_bits)
        
        print(f"Bit allocation: R={self.r_bits}, G={self.g_bits}, B={self.b_bits} bits")
        print(f"Color levels: R={len(self.levels_r)}, G={len(self.levels_g)}, B={len(self.levels_b)}")
    
    def calculate_bit_allocation(self, total_bits):
        """
        Dynamically allocate bits to RGB channels
        Blue gets floor(total_bits/3), Red and Green get ceil(total_bits/3)
        """
        base_bits = total_bits // 3
        remainder = total_bits % 3
        
        if remainder == 0:
            r_bits = base_bits
            g_bits = base_bits
            b_bits = base_bits
        elif remainder == 1:
            r_bits = base_bits + 1
            g_bits = base_bits
            b_bits = base_bits
        else:  # remainder == 2
            r_bits = base_bits + 1
            g_bits = base_bits + 1
            b_bits = base_bits
        
        return r_bits, g_bits, b_bits
    
    def calculate_quantization_levels(self, num_bits):
        """Calculate quantization levels for given bit depth"""
        if num_bits == 0:
            return [0]
        num_levels = 2 ** num_bits
        step = 255 / (num_levels - 1) if num_levels > 1 else 255
        levels = [round(i * step) for i in range(num_levels)]
        return levels
    
    def build_rgb_palette(self):
        """Build RGB palette based on dynamic bit allocation"""
        rgb_palette = {}
        palette_index = 1
        
        # Generate all combinations of R, G, B with their respective bit depths
        for r_level in self.levels_r:
            for g_level in self.levels_g:
                for b_level in self.levels_b:
                    rgb_palette[palette_index] = (b_level, g_level, r_level)  # BGR format for OpenCV
                    palette_index += 1
        
        return rgb_palette
    
    def find_closest_level(self, value, levels):
        """Find the closest level using binary search"""
        if len(levels) == 1:
            return levels[0]
            
        left, right = 0, len(levels) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if levels[mid] == value:
                return levels[mid]
            elif levels[mid] < value:
                left = mid + 1
            else:
                right = mid - 1
        
        # Find closest between levels[left] and levels[right]
        if left >= len(levels):
            return levels[right]
        if right < 0:
            return levels[left]
        
        if abs(value - levels[left]) < abs(value - levels[right]):
            return levels[left]
        else:
            return levels[right]
    
    def find_closest_level_log2(self, value, levels):
        """Find closest level using log2-based approach"""
        if len(levels) == 1:
            return levels[0]
            
        if value <= 0:
            return levels[0]
        if value >= 255:
            return levels[-1]
        
        # Calculate the ideal level based on logarithmic distribution
        log_value = log2(value + 1)  # +1 to avoid log(0)
        max_log = log2(256)  # log2(255+1)
        
        normalized = log_value / max_log
        target_index = int(normalized * (len(levels) - 1))
        
        # Ensure index is within bounds
        target_index = max(0, min(target_index, len(levels) - 1))
        
        return levels[target_index]
    
    def quantize_channel(self, channel, levels, method='binary'):
        """Quantize a single channel to specified levels"""
        quantized = np.zeros_like(channel)
        
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                value = channel[i, j]
                
                if method == 'log2':
                    closest_level = self.find_closest_level_log2(value, levels)
                else:
                    closest_level = self.find_closest_level(value, levels)
                
                quantized[i, j] = closest_level
        
        return quantized
    
    def reduce_to_palette(self, image, method='binary'):
        """
        Reduce image to dynamic bit palette
        Returns: quantized image, palette indices, RGB palette
        """
        # Split channels
        b, g, r = cv2.split(image)
        
        # Quantize each channel
        r_quantized = self.quantize_channel(r, self.levels_r, method)
        g_quantized = self.quantize_channel(g, self.levels_g, method)
        b_quantized = self.quantize_channel(b, self.levels_b, method)
        
        # Merge quantized channels
        quantized_img = cv2.merge([b_quantized, g_quantized, r_quantized])
        
        # Build RGB palette and create index map
        rgb_palette = self.build_rgb_palette()
        index_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Create palette lookup dictionary for faster indexing
        palette_lookup = {}
        for idx, color in rgb_palette.items():
            palette_lookup[color] = idx
        
        # Map each pixel to palette index
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel_color = (int(b_quantized[i, j]), int(g_quantized[i, j]), int(r_quantized[i, j]))
                index_map[i, j] = palette_lookup.get(pixel_color, 1)  # Default to index 1
        
        return quantized_img, index_map, rgb_palette
    
    def reconstruct_from_palette(self, index_map, palette):
        """Reconstruct image from palette indices"""
        height, width = index_map.shape
        reconstructed = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                palette_idx = index_map[i, j]
                if palette_idx in palette:
                    # OpenCV uses BGR format
                    reconstructed[i, j] = palette[palette_idx]
        
        return reconstructed
    
    def calculate_compression_stats(self, original_img, index_map, palette):
        """Calculate compression statistics"""
        original_size = original_img.nbytes  # 24 bits per pixel
        
        # Compressed size: indices (8 bits per pixel) + palette storage
        index_size = index_map.nbytes  # 8 bits per pixel for indices
        palette_size = len(palette) * 3  # 3 bytes per palette color
        
        total_compressed_size = index_size + palette_size
        
        compression_ratio = original_size / total_compressed_size
        bpp_original = 24  # bits per pixel
        bpp_compressed = (index_size * 8 + palette_size * 8) / (original_img.shape[0] * original_img.shape[1])
        
        total_colors = len(self.levels_r) * len(self.levels_g) * len(self.levels_b)
        
        return {
            'original_size': original_size,
            'compressed_size': total_compressed_size,
            'compression_ratio': compression_ratio,
            'bpp_original': bpp_original,
            'bpp_compressed': bpp_compressed,
            'index_size': index_size,
            'palette_size': palette_size,
            'palette_colors': len(palette),
            'total_colors': total_colors,
            'r_bits': self.r_bits,
            'g_bits': self.g_bits,
            'b_bits': self.b_bits
        }


class VisualizationUtils:
    """Utility class for visualization"""
    
    @staticmethod
    def display_comparison(original, compressed, title="Dynamic Palette Compression"):
        """Display original and compressed images"""
        plt.figure(figsize=(12, 6))
        
        # Convert BGR to RGB for display
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title('Original Image (24-bit RGB)')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(compressed_rgb)
        plt.title(title)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def display_palette(palette, title="Color Palette", max_colors=64):
        """Display the color palette"""
        palette_size = len(palette)
        cols = min(8, palette_size)  # Adjust columns based on palette size
        rows = (min(palette_size, max_colors) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))
        fig.suptitle(f'{title} - {palette_size} colors', fontsize=16)
        
        displayed_colors = min(palette_size, max_colors)
        
        for idx, color_idx in enumerate(sorted(palette.keys())[:displayed_colors]):
            row = idx // cols
            col = idx % cols
            
            color_bgr = palette[color_idx]
            color_rgb = (color_bgr[2] / 255.0, color_bgr[1] / 255.0, color_bgr[0] / 255.0)
            
            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col] if cols > 1 else axes
            
            ax.imshow([[color_rgb]])
            ax.set_title(f'#{color_idx}', fontsize=8)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(displayed_colors, rows * cols):
            row = idx // cols
            col = idx % cols
            
            if rows > 1:
                axes[row, col].axis('off')
            else:
                if cols > 1:
                    axes[col].axis('off')
                else:
                    axes.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def display_bit_allocation(r_bits, g_bits, b_bits, total_bits):
        """Display bit allocation as a bar chart"""
        channels = ['Red', 'Green', 'Blue']
        bits = [r_bits, g_bits, b_bits]
        colors = ['red', 'green', 'blue']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(channels, bits, color=colors, alpha=0.7, edgecolor='black')
        
        plt.title(f'Bit Allocation (Total: {total_bits} bits)\nR:{r_bits}, G:{g_bits}, B:{b_bits}')
        plt.ylabel('Number of Bits')
        plt.ylim(0, max(bits) + 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, bits):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value} bits', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def display_quantization_levels(levels_r, levels_g, levels_b):
        """Display quantization levels for each channel"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Red channel levels
        colors_r = [(level/255, 0, 0) for level in levels_r]
        for i, (level, color) in enumerate(zip(levels_r, colors_r)):
            ax1.bar(i, level, color=color, edgecolor='black', alpha=0.7)
            ax1.text(i, level + 5, str(level), ha='center', va='bottom', fontsize=8)
        ax1.set_title(f'Red Channel ({len(levels_r)} levels)')
        ax1.set_xlabel('Level Index')
        ax1.set_ylabel('Intensity Value')
        ax1.set_xticks(range(len(levels_r)))
        ax1.grid(True, alpha=0.3)
        
        # Green channel levels
        colors_g = [(0, level/255, 0) for level in levels_g]
        for i, (level, color) in enumerate(zip(levels_g, colors_g)):
            ax2.bar(i, level, color=color, edgecolor='black', alpha=0.7)
            ax2.text(i, level + 5, str(level), ha='center', va='bottom', fontsize=8)
        ax2.set_title(f'Green Channel ({len(levels_g)} levels)')
        ax2.set_xlabel('Level Index')
        ax2.set_ylabel('Intensity Value')
        ax2.set_xticks(range(len(levels_g)))
        ax2.grid(True, alpha=0.3)
        
        # Blue channel levels
        colors_b = [(0, 0, level/255) for level in levels_b]
        for i, (level, color) in enumerate(zip(levels_b, colors_b)):
            ax3.bar(i, level, color=color, edgecolor='black', alpha=0.7)
            ax3.text(i, level + 5, str(level), ha='center', va='bottom', fontsize=8)
        ax3.set_title(f'Blue Channel ({len(levels_b)} levels)')
        ax3.set_xlabel('Level Index')
        ax3.set_ylabel('Intensity Value')
        ax3.set_xticks(range(len(levels_b)))
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function demonstrating dynamic palette-based color reduction"""
    
    # Load an image
    img_path = '/home/bibrity/CSE_Courses/CSE4161_DIP/Images/paddy_field1.jpeg'
    original_img = cv2.imread(img_path)
    
    # If image not found, create a sample image
    if original_img is None:
        print("Image not found, creating sample image...")
        # Create a colorful sample image
        original_img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(original_img, (0, 0), (100, 200), (255, 0, 0), -1)    # Red
        cv2.rectangle(original_img, (100, 0), (200, 200), (0, 255, 0), -1)  # Green
        cv2.rectangle(original_img, (200, 0), (300, 200), (0, 0, 255), -1)  # Blue
        cv2.circle(original_img, (150, 100), 50, (255, 255, 0), -1)         # Cyan
    
    print(f"Original image shape: {original_img.shape}")
    print(f"Original image size: {original_img.nbytes} bytes")
    
    # Test different total bit values
    total_bits_list = [3, 4, 5, 6, 7, 8]
    methods = ['binary', 'log2']
    
    for total_bits in total_bits_list:
        print(f"\n{'='*60}")
        print(f"TESTING WITH TOTAL BITS: {total_bits}")
        print(f"{'='*60}")
        
        # Initialize color reducer with dynamic bit allocation
        reducer = DynamicPaletteColorReducer(total_bits)
        
        # Display bit allocation
        VisualizationUtils.display_bit_allocation(reducer.r_bits, reducer.g_bits, reducer.b_bits, total_bits)
        
        # Display quantization levels
        VisualizationUtils.display_quantization_levels(reducer.levels_r, reducer.levels_g, reducer.levels_b)
        
        for method in methods:
            print(f"\n--- Using {method.upper()} search method ---")
            
            # Reduce to palette
            quantized_img, index_map, palette = reducer.reduce_to_palette(original_img, method)
            
            # Calculate statistics
            stats = reducer.calculate_compression_stats(original_img, index_map, palette)
            
            print(f"Bit allocation: R={stats['r_bits']}, G={stats['g_bits']}, B={stats['b_bits']}")
            print(f"Total colors: {stats['total_colors']}")
            print(f"Original size: {stats['original_size']} bytes")
            print(f"Compressed size: {stats['compressed_size']} bytes")
            print(f"Compression ratio: {stats['compression_ratio']:.2f}:1")
            print(f"Original BPP: {stats['bpp_original']}")
            print(f"Compressed BPP: {stats['bpp_compressed']:.2f}")
            
            # Reconstruct from palette to verify
            reconstructed_img = reducer.reconstruct_from_palette(index_map, palette)
            
            # Display results
            title = f"Dynamic {total_bits}-bit ({stats['r_bits']}-{stats['g_bits']}-{stats['b_bits']})"
            VisualizationUtils.display_comparison(original_img, reconstructed_img, title)
            
            # Display palette if not too large
            if stats['palette_colors'] <= 256:  # Show palette if reasonable size
                VisualizationUtils.display_palette(palette, f"{total_bits}-bit Palette ({method})")
            
            # Verify reconstruction quality
            mse = np.mean((original_img.astype(float) - reconstructed_img.astype(float)) ** 2)
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
            print(f"Reconstruction quality - MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")


if __name__ == '__main__':
    main()
