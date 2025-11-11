import cv2
import numpy as np
from dahuffman import HuffmanCodec
import pickle
import os
import pywt

class GlobalDWTCompressor:
    def __init__(self, quality=50, wavelet='db4'):
        self.quality = quality
        self.wavelet = wavelet
        self.codec = None
        self.dwt_size = None
        self.coeff_slices = None
        
    def forward_transform(self, image):
        """Apply global DWT transform to the entire image"""
        # Convert to float32 for DWT
        image_float = image.astype(np.float32)
        
        # Apply DWT to each channel if color image
        if len(image.shape) == 3:
            coeffs_list = []
            for i in range(3):
                channel_coeffs = pywt.wavedec2(image_float[:, :, i], self.wavelet, level=3)
                coeffs_list.append(channel_coeffs)
            
            # Flatten the coefficients for easier handling
            flattened_coeffs, coeff_slices = self._flatten_coeffs(coeffs_list)
            self.coeff_slices = coeff_slices
        else:
            # Grayscale image
            coeffs = pywt.wavedec2(image_float, self.wavelet, level=3)
            flattened_coeffs, coeff_slices = self._flatten_coeffs([coeffs])
            self.coeff_slices = coeff_slices
            
        self.dwt_size = flattened_coeffs.shape
        return flattened_coeffs
    
    def inverse_transform(self, dwt_coeffs):
        """Apply inverse DWT transform"""
        # Reconstruct coefficients structure
        if len(self.coeff_slices) == 3:
            # Color image
            reconstructed = np.zeros((dwt_coeffs.shape[0], dwt_coeffs.shape[1], 3), dtype=np.float32)
            for i in range(3):
                coeffs_reconstructed = self._reconstruct_coeffs(dwt_coeffs[:, :, i], self.coeff_slices[i])
                reconstructed[:, :, i] = pywt.waverec2(coeffs_reconstructed, self.wavelet)
        else:
            # Grayscale image
            coeffs_reconstructed = self._reconstruct_coeffs(dwt_coeffs, self.coeff_slices[0])
            reconstructed = pywt.waverec2(coeffs_reconstructed, self.wavelet)
            
        # Clip and convert back to uint8
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        return reconstructed
    
    def _flatten_coeffs(self, coeffs_list):
        """Flatten wavelet coefficients into 2D arrays"""
        flattened_channels = []
        coeff_slices = []
        
        for coeffs in coeffs_list:
            # Get the maximum dimensions needed
            max_rows = 0
            max_cols = 0
            for arr in self._iterate_coeffs(coeffs):
                max_rows = max(max_rows, arr.shape[0])
                max_cols = max(max_cols, arr.shape[1])
            
            # Create a 2D array to store all coefficients
            flattened = np.zeros((max_rows, max_cols), dtype=np.float32)
            current_slice = []
            
            row_pos = 0
            col_pos = 0
            max_col_used = 0
            
            for arr in self._iterate_coeffs(coeffs):
                rows, cols = arr.shape
                flattened[row_pos:row_pos+rows, col_pos:col_pos+cols] = arr
                current_slice.append((row_pos, row_pos+rows, col_pos, col_pos+cols))
                
                col_pos += cols
                max_col_used = max(max_col_used, col_pos)
                
                # Move to next row if we run out of columns
                if col_pos >= max_cols:
                    col_pos = 0
                    row_pos += rows
            
            flattened_channels.append(flattened[:row_pos+rows, :max_col_used])
            coeff_slices.append(current_slice)
        
        # Stack channels for color images
        if len(flattened_channels) == 3:
            flattened_array = np.stack(flattened_channels, axis=-1)
        else:
            flattened_array = flattened_channels[0]
            
        return flattened_array, coeff_slices
    
    def _reconstruct_coeffs(self, flattened, slices):
        """Reconstruct wavelet coefficients structure from flattened array"""
        coeffs = []
        slice_idx = 0
        
        # Reconstruct the nested coefficient structure
        # For 3-level decomposition: [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]
        level3_slices = slices[:4]  # cA3, cH3, cV3, cD3
        level2_slices = slices[4:7] # cH2, cV2, cD2
        level1_slices = slices[7:]  # cH1, cV1, cD1
        
        # Level 3 coefficients
        cA3 = self._extract_slice(flattened, level3_slices[0])
        cH3 = self._extract_slice(flattened, level3_slices[1])
        cV3 = self._extract_slice(flattened, level3_slices[2])
        cD3 = self._extract_slice(flattened, level3_slices[3])
        
        # Level 2 coefficients
        cH2 = self._extract_slice(flattened, level2_slices[0])
        cV2 = self._extract_slice(flattened, level2_slices[1])
        cD2 = self._extract_slice(flattened, level2_slices[2])
        
        # Level 1 coefficients
        cH1 = self._extract_slice(flattened, level1_slices[0])
        cV1 = self._extract_slice(flattened, level1_slices[1])
        cD1 = self._extract_slice(flattened, level1_slices[2])
        
        return [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]
    
    def _extract_slice(self, flattened, slice_info):
        """Extract a subarray from flattened coefficients"""
        r_start, r_end, c_start, c_end = slice_info
        return flattened[r_start:r_end, c_start:c_end]
    
    def _iterate_coeffs(self, coeffs):
        """Iterate through all coefficient arrays in the wavelet decomposition"""
        yield coeffs[0]  # Approximation coefficients
        for detail_tuple in coeffs[1:]:
            for detail_coeff in detail_tuple:
                yield detail_coeff
    
    def quantizer(self, dwt_coeffs):
        """Uniform quantization of DWT coefficients"""
        # Calculate quantization step based on quality
        quality_factor = max(1, min(100, self.quality))
        
        # Map quality (1-100) to quantization step
        # More aggressive quantization for higher compression
        quantization_step = 100 - quality_factor
        quantization_step = max(1, quantization_step)
        
        print(f"Using uniform quantization step: {quantization_step:.2f}")
        
        # Apply uniform quantization
        quantized = np.round(dwt_coeffs / quantization_step).astype(np.int32)
        
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
        print("Applying forward DWT transform...")
        dwt_coeffs = self.forward_transform(image)
        
        print("Quantizing coefficients...")
        quantized_coeffs, quantization_step = self.quantizer(dwt_coeffs)
        
        print("Huffman encoding...")
        encoded_data = self.symbol_encoder(quantized_coeffs)
        
        # Prepare compressed data package
        compressed_data = {
            'encoded_data': encoded_data,
            'quantization_step': quantization_step,
            'shape': quantized_coeffs.shape,
            'original_shape': image.shape,
            'quality': self.quality,
            'wavelet': self.wavelet,
            'coeff_slices': self.coeff_slices
        }
        
        return compressed_data
    
    def decompress(self, compressed_data):
        """Complete decompression pipeline"""
        self.wavelet = compressed_data.get('wavelet', 'db4')
        self.coeff_slices = compressed_data['coeff_slices']
        
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
        
        print("Applying inverse DWT transform...")
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

def compare_dct_dwt():
    """Compare DCT and DWT compression performance"""
    # Load an image
    image_path = "input_image.jpg"
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Using a sample image...")
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite("sample_image.jpg", image)
        image_path = "sample_image.jpg"
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    # Convert BGR to RGB for better visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Original image shape: {image_rgb.shape}")
    print(f"Original image size: {image_rgb.nbytes} bytes")
    
    # Test both methods
    methods = [
        ('DCT', GlobalDCTCompressor(quality=75)),
        ('DWT', GlobalDWTCompressor(quality=75, wavelet='db4'))
    ]
    
    for method_name, compressor in methods:
        print(f"\n{'='*60}")
        print(f"Testing {method_name} Compression")
        print(f"{'='*60}")
        
        # Compression
        print(f"\n=== {method_name} Compression Phase ===")
        compressed_data = compressor.compress(image_rgb)
        
        # Save compressed data
        save_compressed_data(compressed_data, f"compressed_{method_name.lower()}.pkl")
        
        # Decompression
        print(f"\n=== {method_name} Decompression Phase ===")
        compressed_data = load_compressed_data(f"compressed_{method_name.lower()}.pkl")
        decompressed_image = compressor.decompress(compressed_data)
        
        # Calculate metrics
        compression_ratio, original_size, compressed_size = calculate_compression_ratio(
            image_rgb, compressed_data
        )
        psnr_value = calculate_psnr(image_rgb, decompressed_image)
        
        print(f"\n=== {method_name} Results ===")
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}:1")
        print(f"PSNR: {psnr_value:.2f} dB")
        
        # Save decompressed image
        cv2.imwrite(f"decompressed_{method_name.lower()}.jpg", 
                   cv2.cvtColor(decompressed_image, cv2.COLOR_RGB2BGR))
        print(f"Decompressed image saved as 'decompressed_{method_name.lower()}.jpg'")

# Example usage with different wavelets
def test_different_wavelets():
    """Test DWT compression with different wavelet types"""
    image_path = "input_image.jpg"
    if not os.path.exists(image_path):
        return
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    wavelets = ['db1', 'db4', 'db8', 'haar', 'bior2.2', 'bior4.4']
    quality = 75
    
    for wavelet in wavelets:
        print(f"\nTesting wavelet: {wavelet}")
        try:
            compressor = GlobalDWTCompressor(quality=quality, wavelet=wavelet)
            compressed_data = compressor.compress(image_rgb)
            decompressed_image = compressor.decompress(compressed_data)
            
            psnr_value = calculate_psnr(image_rgb, decompressed_image)
            compression_ratio, _, _ = calculate_compression_ratio(image_rgb, compressed_data)
            
            print(f"Wavelet: {wavelet}, PSNR: {psnr_value:.2f} dB, Ratio: {compression_ratio:.2f}:1")
            
        except Exception as e:
            print(f"Failed with wavelet {wavelet}: {e}")

if __name__ == "__main__":
    # Compare DCT vs DWT
    compare_dct_dwt()
    
    # Test different wavelets
    print("\n\nTesting different wavelets:")
    test_different_wavelets()
