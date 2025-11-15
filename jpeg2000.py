import numpy as np
import pywt
import cv2

def im2jpeg2k(x, n, q):
    """
    Compresses an image using JPEG2000-like compression
    
    Parameters:
    x: input image (uint8)
    n: number of wavelet decomposition levels
    q: quantization step size vector
    """
    # Check input arguments
    if not isinstance(x, np.ndarray) or not np.isreal(x).all() or x.dtype != np.uint8:
        raise ValueError('The input must be a UINT8 image.')
    
    if len(q) != 2 and len(q) != 3 * n + 1:
        raise ValueError('The quantization step size vector is bad.')
    
    # Level shift the input and compute its wavelet transform
    x = x.astype(np.float64) - 128
    
    # Compute wavelet transform using 9/7 filter (bior4.4 is similar to JPEG 9/7)
    coeffs = pywt.wavedec2(x, 'bior4.4', level=n)
    
    # Convert coefficients to single array using PyWavelets built-in function
    c, s = pywt.coeffs_to_array(coeffs)
    
    # Quantize the wavelet coefficients
    q_steps = stepsize(n, q)
    
    # Apply quantization to coefficients
    c_quantized = quantize_coefficients(c, coeffs, n, q_steps)
    
    # Run-length code zero runs of more than 10
    encoded_data = runlength_encode(c_quantized)
    
    # Prepare output structure
    y = {
        'runs': np.uint16(encoded_data['runs_table']),
        's': np.uint16(s),  # s is already a flat array from pywt.coeffs_to_array
        'zrc': np.uint16(-encoded_data['zrc']),
        'q': np.uint16(np.round(100 * np.array(q_steps))),
        'n': np.uint16(n),
        'huffman': encoded_data['encoded_data']
    }
    
    return y

def quantize_coefficients(c, coeffs, n, q_steps):
    """Quantize wavelet coefficients using subband-specific steps"""
    # Get the coefficient structure to understand subband organization
    coeffs_arr, coeff_slices, coeff_shapes = pywt.coeffs_to_array(coeffs, padding=0)
    
    # Create a mask for quantization steps
    q_mask = np.zeros_like(c, dtype=np.float64)
    
    # Apply quantization steps based on subband type and level
    current_idx = 0
    
    # Approximation coefficients (lowest frequency - finest quantization)
    app_coeffs = coeffs[0]
    app_size = app_coeffs.size
    q_mask[current_idx:current_idx + app_size] = q_steps[-1]
    current_idx += app_size
    
    # Detail coefficients for each level (from coarsest to finest)
    for level in range(n, 0, -1):
        # For each subband type: horizontal, vertical, diagonal
        for subband_idx, subband_type in enumerate(['h', 'v', 'd']):
            if level <= len(coeffs) - 1:
                if subband_idx == 0:  # horizontal
                    coeff = coeffs[level][0]
                elif subband_idx == 1:  # vertical  
                    coeff = coeffs[level][1]
                else:  # diagonal
                    coeff = coeffs[level][2]
                
                coeff_size = coeff.size
                qi = 3 * (n - level) + subband_idx
                q_mask[current_idx:current_idx + coeff_size] = q_steps[qi]
                current_idx += coeff_size
    
    # Apply quantization
    c_abs = np.abs(c)
    sgn = np.sign(c)
    sgn[sgn == 0] = 1
    
    # Quantize and restore signs
    c_quantized = np.floor(c_abs / q_mask + 0.5)
    c_quantized = c_quantized * sgn
    
    return c_quantized.astype(np.int32)

def runlength_encode(c):
    """Run-length encode zero runs of more than 10"""
    zrc = np.min(c) - 1  # Special code for zero run
    eoc = zrc - 1        # End of code
    
    runs_table = [65535]  # Shared runs table
    
    # Find zero runs using difference method
    z = (c == 0).astype(np.int32)
    z_diff = np.diff(np.concatenate(([0], z, [0])))
    plus = np.where(z_diff == 1)[0]   # Start of zero runs
    minus = np.where(z_diff == -1)[0] # End of zero runs + 1
    
    # Remove any terminating zero run
    if len(plus) != len(minus):
        c_modified = np.concatenate((c[:plus[-1]], [eoc]))
        plus = plus[:-1]
        minus = minus[:-1]
    else:
        c_modified = c.copy()
    
    # Process zero runs from end to beginning
    encoded = c_modified.tolist()
    
    for i in range(len(minus) - 1, -1, -1):
        run_length = minus[i] - plus[i]
        if run_length > 10:
            # Remove the original run
            encoded = encoded[:plus[i]] + encoded[minus[i]:]
            
            # Encode the run length
            overflow = run_length // 65535
            remaining_run = run_length - overflow * 65535
            
            # Add encoded run
            run_encoding = []
            for _ in range(overflow):
                run_encoding.extend([zrc, 1])  # zrc + overflow marker
            
            run_encoding.append(zrc)
            run_idx = find_or_add_run(runs_table, remaining_run)
            run_encoding.append(run_idx)
            
            encoded = encoded[:plus[i]] + run_encoding + encoded[plus[i]:]
    
    return {
        'encoded_data': encoded,
        'runs_table': runs_table,
        'zrc': zrc
    }

def find_or_add_run(runs_table, run_length):
    """Find a zero run in the run-length table or create new entry"""
    if run_length in runs_table:
        return runs_table.index(run_length) + 1  # 1-based indexing
    else:
        runs_table.append(run_length)
        return len(runs_table)  # Return new index

def stepsize(n, p):
    """Create subband quantization array of step sizes"""
    if len(p) == 2:  # Implicit quantization
        q = []
        qn = 2 ** (8 - p[1] + n) * (1 + p[0] / 2 ** 11)
        
        for k in range(1, n + 1):
            qk = 2 ** -k * qn
            # For each level: horizontal, vertical, diagonal
            q.extend([2 * qk, 2 * qk, 4 * qk])
        
        q.append(qk)  # Approximation subband (finest quantization)
    else:  # Explicit quantization
        q = p
    
    # Round to 1/100th place
    q = np.round(np.array(q) * 100) / 100
    
    if np.any(100 * q > 65535):
        raise ValueError('The quantizing steps are not UINT16 representable.')
    
    if np.any(q == 0):
        raise ValueError('A quantizing step of 0 is not allowed.')
    
    return q

def jpeg2k2im(y):
    """Decode JPEG2000-like compressed image"""
    # Extract parameters
    n = int(y['n'])
    q_steps = y['q'].astype(np.float64) / 100
    s = y['s']  # Shape information from pywt.coeffs_to_array
    zrc = -int(y['zrc'])
    runs_table = y['runs'].tolist()
    encoded_data = y['huffman']
    
    # Decode run-length encoding
    decoded_data = runlength_decode(encoded_data, zrc, runs_table)
    
    # Dequantize coefficients
    c_dequantized = dequantize_coefficients(decoded_data, n, q_steps, s)
    
    # Inverse wavelet transform
    coeffs = pywt.array_to_coeffs(c_dequantized, s, output_format='wavedec2')
    x_reconstructed = pywt.waverec2(coeffs, 'bior4.4')
    
    # Level shift back and clip to valid range
    x_reconstructed = np.clip(x_reconstructed + 128, 0, 255).astype(np.uint8)
    
    return x_reconstructed

def runlength_decode(encoded_data, zrc, runs_table):
    """Decode run-length encoded data"""
    decoded = []
    i = 0
    
    while i < len(encoded_data):
        if encoded_data[i] == zrc:
            # Zero run encountered
            i += 1
            if i < len(encoded_data) and encoded_data[i] == 1:
                # Overflow run
                run_length = 65535
                i += 1
                # Check for additional overflow runs
                while i < len(encoded_data) and encoded_data[i] == zrc and i+1 < len(encoded_data) and encoded_data[i+1] == 1:
                    run_length += 65535
                    i += 2
            else:
                # Normal run
                run_idx = encoded_data[i] - 1  # Convert to 0-based index
                run_length = runs_table[run_idx]
                i += 1
            
            # Add zero run
            decoded.extend([0] * run_length)
        else:
            decoded.append(encoded_data[i])
            i += 1
    
    return np.array(decoded)

def dequantize_coefficients(c, n, q_steps, s):
    """Dequantize wavelet coefficients"""
    # Reconstruct the coefficient array structure for dequantization
    coeffs = pywt.array_to_coeffs(c, s, output_format='wavedec2')
    coeffs_arr, coeff_slices, coeff_shapes = pywt.coeffs_to_array(coeffs, padding=0)
    
    # Create quantization mask (same as in quantization)
    q_mask = np.zeros_like(c, dtype=np.float64)
    current_idx = 0
    
    # Approximation coefficients
    app_coeffs = coeffs[0]
    app_size = app_coeffs.size
    q_mask[current_idx:current_idx + app_size] = q_steps[-1]
    current_idx += app_size
    
    # Detail coefficients for each level
    for level in range(n, 0, -1):
        for subband_idx, subband_type in enumerate(['h', 'v', 'd']):
            if level <= len(coeffs) - 1:
                if subband_idx == 0:  # horizontal
                    coeff = coeffs[level][0]
                elif subband_idx == 1:  # vertical  
                    coeff = coeffs[level][1]
                else:  # diagonal
                    coeff = coeffs[level][2]
                
                coeff_size = coeff.size
                qi = 3 * (n - level) + subband_idx
                q_mask[current_idx:current_idx + coeff_size] = q_steps[qi]
                current_idx += coeff_size
    
    # Apply dequantization
    c_abs = np.abs(c)
    sgn = np.sign(c)
    sgn[sgn == 0] = 1
    
    c_dequantized = c_abs * q_mask
    c_dequantized = c_dequantized * sgn
    
    return c_dequantized

# Example usage with improved testing
if __name__ == "__main__":
    # Read an image or create test image
    img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Creating test image...")
        # Create a more meaningful test image
        x, y = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 255, 256))
        img = (128 + 127 * np.sin(0.1 * x) * np.cos(0.1 * y)).astype(np.uint8)
    
    print(f"Original image size: {img.shape}")
    print(f"Image data type: {img.dtype}")
    
    # Compress the image
    n_levels = 3
    q_vector = [0.5, 8]  # Implicit quantization parameters
    
    try:
        compressed = im2jpeg2k(img, n_levels, q_vector)
        print("Compression completed successfully")
        
        # Decompress the image
        reconstructed = jpeg2k2im(compressed)
        
        # Calculate PSNR
        mse = np.mean((img.astype(float) - reconstructed.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        print(f"PSNR: {psnr:.2f} dB")
        
        # Display results
        cv2.imshow('Original', img)
        cv2.imshow('Reconstructed', reconstructed)
        print("Press any key to close images...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
