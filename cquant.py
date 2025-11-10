import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

class ColorMapQuantizer:
    def __init__(self):
        self.color_map = None
        self.palette = None
        self.quantized_image = None
        
    def create_color_map(self, map_type: str = 'rainbow') -> Callable[[float], Tuple[int, int, int]]:
        """
        Create a color map function φ: [0,1] → C (RGB color space)
        
        Args:
            map_type: Type of color map ('rainbow', 'heat', 'cool', 'grayscale')
        
        Returns:
            Color map function that takes t in [0,1] and returns RGB color
        """
        if map_type == 'rainbow':
            def rainbow_map(t: float) -> Tuple[int, int, int]:
                # Rainbow color map: red -> yellow -> green -> cyan -> blue -> magenta
                r = int(255 * (1 - abs(2 * t - 1)))
                g = int(255 * (1 - abs(2 * t - 0.5)))
                b = int(255 * (1 - abs(2 * t - 1.5)))
                return (b, g, r)  # OpenCV uses BGR
            return rainbow_map
            
        elif map_type == 'heat':
            def heat_map(t: float) -> Tuple[int, int, int]:
                # Heat map: black -> red -> yellow -> white
                if t < 0.33:
                    r = int(255 * (t / 0.33))
                    return (0, 0, r)
                elif t < 0.66:
                    r = 255
                    g = int(255 * ((t - 0.33) / 0.33))
                    return (0, g, r)
                else:
                    r = g = 255
                    b = int(255 * ((t - 0.66) / 0.34))
                    return (b, g, r)
            return heat_map
            
        elif map_type == 'grayscale':
            def grayscale_map(t: float) -> Tuple[int, int, int]:
                # Grayscale map: black -> white
                intensity = int(255 * t)
                return (intensity, intensity, intensity)
            return grayscale_map
            
        else:
            raise ValueError(f"Unknown color map type: {map_type}")
    
    def create_uniform_palette(self, n: int, color_map: Callable) -> List[Tuple[int, int, int]]:
        """
        Create a uniform palette by discretizing [0,1] into n subintervals
        
        Args:
            n: Number of quantization levels
            color_map: Color map function φ: [0,1] → RGB
        
        Returns:
            List of n colors representing the quantized palette
        """
        palette = []
        # Uniform partition of [0,1]
        t_values = np.linspace(0, 1, n)
        
        for t in t_values:
            color = color_map(t)
            palette.append(color)
            
        return palette
    
    def apply_color_map_quantization(self, image: np.ndarray, n_bits: int, 
                                   color_map_type: str = 'rainbow') -> np.ndarray:
        """
        Apply color map quantization to an image
        
        Args:
            image: Input image (BGR format)
            n_bits: Number of bits for quantization (determines palette size n = 2^n_bits)
            color_map_type: Type of color map to use
        
        Returns:
            Quantized image
        """
        # Convert image to grayscale for intensity values
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to [0,1] range
        normalized = gray.astype(np.float32) / 255.0
        
        # Create color map and palette
        color_map_func = self.create_color_map(color_map_type)
        n_levels = 2 ** n_bits
        self.palette = self.create_uniform_palette(n_levels, color_map_func)
        
        # Quantize the normalized values
        quantized = np.zeros_like(image)
        height, width = normalized.shape
        
        for i in range(height):
            for j in range(width):
                t = normalized[i, j]
                # Find closest palette color
                palette_index = int(t * (n_levels - 1))
                palette_index = max(0, min(n_levels - 1, palette_index))
                quantized[i, j] = self.palette[palette_index]
        
        self.quantized_image = quantized.astype(np.uint8)
        return self.quantized_image
    
    def apply_direct_quantization(self, image: np.ndarray, n_bits: int) -> np.ndarray:
        """
        Apply direct uniform quantization to RGB channels (for comparison)
        
        Args:
            image: Input image
            n_bits: Number of bits per channel
        
        Returns:
            Quantized image
        """
        n_levels = 2 ** n_bits
        quantization_step = 256 // n_levels
        
        # Apply uniform quantization to each channel
        quantized = (image // quantization_step) * quantization_step
        return quantized
    
    def visualize_results(self, original: np.ndarray, quantized: np.ndarray, 
                         palette: List[Tuple[int, int, int]], title: str):
        """
        Visualize original image, quantized image, and color palette
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Quantized image
        axes[1].imshow(cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Quantized Image\n{title}')
        axes[1].axis('off')
        
        # Color palette
        self._plot_palette(axes[2], palette)
        axes[2].set_title('Color Palette')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_palette(self, ax, palette: List[Tuple[int, int, int]]):
        """Plot the color palette as a horizontal bar"""
        n_colors = len(palette)
        palette_array = np.zeros((50, n_colors, 3), dtype=np.uint8)
        
        for i, color in enumerate(palette):
            palette_array[:, i, :] = color
        
        # Convert BGR to RGB for display
        palette_rgb = cv2.cvtColor(palette_array, cv2.COLOR_BGR2RGB)
        ax.imshow(palette_rgb)
        ax.set_xlabel(f'{n_colors} colors')

def main():
    # Create or load a sample image
    print("Creating sample image...")
    
    # Create a gradient test image
    height, width = 300, 400
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create smooth gradient from black to white
    for i in range(height):
        for j in range(width):
            intensity = int(255 * (j / width))
            gradient[i, j] = [intensity, intensity, intensity]
    
    # Add some color variation
    for i in range(height):
        for j in range(width):
            # Add subtle color variations
            r_var = int(50 * np.sin(2 * np.pi * i / 100))
            g_var = int(50 * np.cos(2 * np.pi * j / 150))
            gradient[i, j, 2] = min(255, max(0, gradient[i, j, 2] + r_var))  # Red channel
            gradient[i, j, 1] = min(255, max(0, gradient[i, j, 1] + g_var))  # Green channel
    
    original_image = gradient
    
    quantizer = ColorMapQuantizer()
    
    print("Testing different quantization methods...")
    
    # Test different bit depths with rainbow color map
    bit_depths = [8, 4, 2]  # 256, 16, 4 colors respectively
    
    for n_bits in bit_depths:
        print(f"\nQuantizing with {n_bits} bits ({2**n_bits} colors)")
        
        # Color map quantization
        quantized_cm = quantizer.apply_color_map_quantization(
            original_image, n_bits, 'rainbow'
        )
        
        # Direct quantization for comparison
        quantized_direct = quantizer.apply_direct_quantization(original_image, n_bits)
        
        # Visualize results
        quantizer.visualize_results(
            original_image, 
            quantized_cm, 
            quantizer.palette,
            f'Color Map Quantization ({n_bits} bits)'
        )
        
        # Show direct quantization for comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(quantized_direct, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Direct Quantization ({n_bits} bits)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Test different color maps with fixed bit depth
    print("\nTesting different color maps with 4-bit quantization:")
    color_maps = ['rainbow', 'heat', 'grayscale']
    
    for cm_type in color_maps:
        print(f"Color map: {cm_type}")
        quantized = quantizer.apply_color_map_quantization(original_image, 4, cm_type)
        quantizer.visualize_results(
            original_image, quantized, quantizer.palette,
            f'{cm_type.capitalize()} Color Map (4 bits)'
        )

if __name__ == "__main__":
    main()
