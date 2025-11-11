import cv2
import numpy as np
from dahuffman import HuffmanCodec
import pickle
import os

class GlobalDCTCompressor:
    def __init__(self, quality=50):
        self.quality = quality
        self.codec = None
        self.dct_size = None
        
    def forward_transform(self, image):
        """Apply global DCT transform to the entire image"""
        # Convert to float32 for DCT
        image_float = image.astype(np.float32)
        
        
        dct_coeffs = cv2.dct(image_float)
            
        self.dct_size = dct_coeffs.shape
        return dct_coeffs
    
    def inverse_transform(self, dct_coeffs):
        """Apply inverse DCT transform"""
        # Apply inverse DCT to each channel
    
        reconstructed = cv2.idct(dct_coeffs)
            
        # Convert back to uint8
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        return reconstructed
    
    def quantizer(self, dct_coeffs):
        """Uniform quantization of DCT coefficients"""
        # Calculate quantization step based on quality
        # Higher quality = smaller step size = less quantization
        quality_factor = max(1, min(100, self.quality))
        
        # Map quality (1-100) to quantization step (50-1)
        quantization_step = 51 - (quality_factor * 0.5)
        quantization_step = max(1, quantization_step)
        
        print(f"Using uniform quantization step: {quantization_step:.2f}")
        
        # Apply uniform quantization
        quantized = np.round(dct_coeffs / quantization_step).astype(np.int32)
        
        return quantized, quantization_step
    
    def dequantizer(self, quantized_coeffs, quantization_step):
        """Dequantize coefficients using uniform quantization"""
        dequantized = quantized_coeffs.astype(np.float32) * quantization_step
        return dequantized
    
    def symbol_encoder(self, quantized_coeffs):
        """Encode quantized coefficients using Huffman coding"""
        # Flatten the array for Huffman coding
        flattened = quantized_coeffs.flatten()
        
        # Create Huffman codec
        self.codec = HuffmanCodec.from_data(flattened)
        
        # Encode the data
        encoded_data = self.codec.encode(flattened)
        
        return encoded_data
    
    def symbol_decoder(self, encoded_data, shape):
        """Decode Huffman encoded data back to quantized coefficients"""
        if self.codec is None:
            raise ValueError("Huffman codec not initialized")
        
        # Decode the data
        decoded_flat = self.codec.decode(encoded_data)
        
        # Reshape to original dimensions
        quantized_coeffs = np.array(decoded_flat).reshape(shape)
        
        return quantized_coeffs
    
    def compress(self, image):
        """Complete compression pipeline"""
        print("Applying forward DCT transform...")
        dct_coeffs = self.forward_transform(image)
        
        print("Quantizing coefficients...")
        quantized_coeffs, quantization_step = self.quantizer(dct_coeffs)
        
        print("Huffman encoding...")
        encoded_data = self.symbol_encoder(quantized_coeffs)
        
        # Prepare compressed data package
        compressed_data = {
            'encoded_data': encoded_data,
            'quantization_step': quantization_step,
            'shape': quantized_coeffs.shape,
            'original_shape': image.shape,
            'quality': self.quality
        }
        
        return compressed_data
    
    def decompress(self, compressed_data):
        """Complete decompression pipeline"""
        print("Huffman decoding...")
        quantized_coeffs = self.symbol_decoder(
            compressed_data['encoded_data'], 
            compressed_data['shape']
        )
        
        print("Dequantizing coefficients...")
        dequantized_coeffs = self.dequantizer(
            quantized_coeffs, 
            compressed_data['quantization_step']
        )
        
        print("Applying inverse DCT transform...")
        reconstructed_image = self.inverse_transform(dequantized_coeffs)
        
        return reconstructed_image

def save_compressed_data(compressed_data, filename):
    """Save compressed data to file"""
    with open(filename, 'wb') as f:
        pickle.dump(compressed_data, f)
    print(f"Compressed data saved to {filename}")

def load_compressed_data(filename):
    """Load compressed data from file"""
    with open(filename, 'rb') as f:
        compressed_data = pickle.load(f)
    return compressed_data

def calculate_compression_ratio(original_image, compressed_data):
    """Calculate compression ratio"""
    original_size = original_image.nbytes
    # For uniform quantization, we only store the step size (single value)
    compressed_size = len(compressed_data['encoded_data']) + 4  # +4 bytes for quantization_step
    
    ratio = original_size / compressed_size
    return ratio, original_size, compressed_size

def calculate_psnr(original, reconstructed):
    """Calculate PSNR between original and reconstructed images"""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Example usage and demonstration
def main():
    # Load an image
    image_path = "input_image.jpg"  # Replace with your image path
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Using a sample image...")
        # Create a sample image if file doesn't exist
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite("sample_image.jpg", image)
        image_path = "sample_image.jpg"
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    # Convert BGR to RGB for better visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Original image shape: {image_rgb.shape}")
    print(f"Original image size: {image_rgb.nbytes} bytes")
    
    # Test different quality levels
    quality_levels = [25, 50, 75, 95]
    
    for quality in quality_levels:
        print(f"\n{'='*50}")
        print(f"Testing with quality: {quality}")
        print(f"{'='*50}")
        
        # Initialize compressor with desired quality
        compressor = GlobalDCTCompressor(quality=quality)
        
        # Compression
        print("\n=== Compression Phase ===")
        compressed_data = compressor.compress(image_rgb)
        
        # Save compressed data
        save_compressed_data(compressed_data, f"compressed_image_q{quality}.pkl")
        
        # Decompression
        print("\n=== Decompression Phase ===")
        compressed_data = load_compressed_data(f"compressed_image_q{quality}.pkl")
        decompressed_image = compressor.decompress(compressed_data)
        
        # Calculate metrics
        compression_ratio, original_size, compressed_size = calculate_compression_ratio(
            image_rgb, compressed_data
        )
        psnr_value = calculate_psnr(image_rgb, decompressed_image)
        
        print(f"\n=== Results ===")
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}:1")
        print(f"PSNR: {psnr_value:.2f} dB")
        
        # Save decompressed image
        cv2.imwrite(f"decompressed_image_q{quality}.jpg", 
                   cv2.cvtColor(decompressed_image, cv2.COLOR_RGB2BGR))
        print(f"Decompressed image saved as 'decompressed_image_q{quality}.jpg'")

if __name__ == "__main__":
    main()
