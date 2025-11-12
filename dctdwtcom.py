import cv2, numpy as np, heapq
from collections import Counter
import matplotlib.pyplot as plt

# ====================== SETTINGS ===========================================
IMG_PATH	= "/home/alamin/1.PART_IV/DIP/images/raw.png"
DCT_QSTEP	= 10.0 
DWT_QSTEP	= 1.0  
WAVELET		= "haar"
# Add thresholding settings
KEEP_RATIOS = [1.0, 0.1, 0.05, 0.01, 0.005]  # 1.0 means no thresholding
# ===========================================================================

# Optional PyWavelets for DWT
try:
	import pywt
	HAS_PYWT = True
except ImportError:
	HAS_PYWT = False

def huff_bits(arr: np.ndarray) -> int:
	flat = arr.ravel().tolist()
	if not flat: return 0
	freq = Counter(flat)
	h=[]; uid=0
	for s,f in freq.items():
		heapq.heappush(h, (f, uid, (s,None,None))); uid+=1
	while len(h)>1:
		f1,_,n1 = heapq.heappop(h)
		f2,_,n2 = heapq.heappop(h)
		heapq.heappush(h, (f1+f2, uid, (None,n1,n2))); uid+=1
	root = h[0][2]
	codes={}
	def walk(node,p=""):
		s,l,r = node
		if s is not None: codes[s]=p or "0"; return
		walk(l,p+"0"); walk(r,p+"1")
	walk(root)
	return sum(len(codes[v]) for v in flat)

def pct_smaller(orig_bits, comp_bits):
	return max(0.0, 100.0*(1.0 - comp_bits/max(1,orig_bits)))

def zigzag_scan(matrix):
    """Zigzag scan of a 2D matrix"""
    rows, cols = matrix.shape
    result = []
    
    # Directions: right, down-left, down, up-right
    directions = [(0, 1), (1, -1), (1, 0), (-1, 1)]
    dir_idx = 0
    i, j = 0, 0
    
    for _ in range(rows * cols):
        result.append(matrix[i, j])
        
        if dir_idx == 0:  # Moving right
            if i == 0:
                dir_idx = 1  # Go down-left
            else:
                dir_idx = 3  # Go up-right
        elif dir_idx == 1:  # Moving down-left
            if i == rows - 1 or j == 0:
                if j == 0:
                    dir_idx = 2  # Go down
                else:
                    dir_idx = 0  # Go right
        elif dir_idx == 2:  # Moving down
            if j == 0:
                dir_idx = 3  # Go up-right
            else:
                dir_idx = 1  # Go down-left
        elif dir_idx == 3:  # Moving up-right
            if j == cols - 1:
                dir_idx = 2  # Go down
            elif i == 0:
                dir_idx = 0  # Go right
        
        # Move to next position based on current direction
        di, dj = directions[dir_idx]
        i += di
        j += dj
    
    return np.array(result)

def run_length_encode(arr):
    """Run-length encoding of an array"""
    if len(arr) == 0:
        return []
    
    encoded = []
    count = 1
    current = arr[0]
    
    for i in range(1, len(arr)):
        if arr[i] == current:
            count += 1
        else:
            encoded.append((current, count))
            current = arr[i]
            count = 1
    
    encoded.append((current, count))
    return encoded

def apply_thresholding(coeffs, keep_ratio):
    """Apply thresholding to DWT coefficients based on keep ratio"""
    if keep_ratio >= 1.0:
        return coeffs  # No thresholding
        
    # For DWT coefficients structure (cA, (cH, cV, cD))
    coeff_flat = []
    # For DWT coefficients structure
    coeff_flat.append(coeffs[0].ravel())  # cA
    for detail_coeff in coeffs[1]:
        coeff_flat.append(detail_coeff.ravel())
    all_coeffs = np.concatenate(coeff_flat)
    
    # Sort by absolute value and find threshold
    abs_coeffs = np.abs(all_coeffs)
    sorted_indices = np.argsort(abs_coeffs)
    thresh_idx = int(len(abs_coeffs) * (1 - keep_ratio))
    
    if thresh_idx > 0:
        threshold = abs_coeffs[sorted_indices[thresh_idx]]
    else:
        threshold = 0
    
    # Apply threshold to DWT coefficients
    start_idx = 0
    new_cA = coeffs[0].copy()
    new_details = []
    
    # Threshold approximation coefficients
    cA_size = coeffs[0].size
    cA_flat = all_coeffs[start_idx:start_idx + cA_size]
    cA_flat[np.abs(cA_flat) < threshold] = 0
    new_cA = cA_flat.reshape(coeffs[0].shape)
    start_idx += cA_size
    
    # Threshold detail coefficients
    for i, detail_coeff in enumerate(coeffs[1]):
        detail_size = detail_coeff.size
        detail_flat = all_coeffs[start_idx:start_idx + detail_size]
        detail_flat[np.abs(detail_flat) < threshold] = 0
        new_details.append(detail_flat.reshape(detail_coeff.shape))
        start_idx += detail_size
        
    return (new_cA, tuple(new_details))

# 1) Huffman (lossless)
def do_huffman(img):
	bits = huff_bits(img)
	return bits, img.copy()

# 2) DCT (cv2) + uniform quant + Zigzag + RLE + Huffman
def do_dct(img, qstep=DCT_QSTEP, keep_ratio=1.0):
	x = img.astype(np.float32) - 128.0
	C = cv2.dct(x)
	Cq = np.rint(C / qstep).astype(np.int32)
	
	# Apply zigzag scan and run-length encoding
	zigzag_coeffs = zigzag_scan(Cq)
	rle_pairs = run_length_encode(zigzag_coeffs)
	
	# Calculate bits from RLE pairs
	# We need to flatten the RLE pairs for Huffman coding
	if rle_pairs:
		# Convert RLE pairs to flat array: [value1, run1, value2, run2, ...]
		flat_rle = []
		for value, run in rle_pairs:
			flat_rle.extend([value, run])
		bits = huff_bits(np.array(flat_rle))
	else:
		bits = 0
	
	y = cv2.idct(Cq.astype(np.float32) * qstep) + 128.0
	return bits, np.clip(y, 0, 255).astype(np.uint8)

# 3) DWT (pywt) + uniform quant + Thresholding + Huffman
def do_dwt(img, wave=WAVELET, qstep=DWT_QSTEP, keep_ratio=1.0):
	if not HAS_PYWT:
		raise RuntimeError("PyWavelets not installed. pip install pywavelets")
	x = img.astype(np.float32)
	coeffs = pywt.dwt2(x, wave)
	cA, (cH, cV, cD) = coeffs
	
	cAq = np.rint(cA / qstep).astype(np.int32)
	cHq = np.rint(cH / qstep).astype(np.int32)
	cVq = np.rint(cV / qstep).astype(np.int32)
	cDq = np.rint(cD / qstep).astype(np.int32)
	
	coeffs_q = (cAq, (cHq, cVq, cDq))
	
	# Apply thresholding only for DWT
	if keep_ratio < 1.0:
		coeffs_q = apply_thresholding(coeffs_q, keep_ratio)
	
	# Calculate bits from thresholded coefficients
	cAq_t, (cHq_t, cVq_t, cDq_t) = coeffs_q
	bits = huff_bits(np.concatenate([cAq_t.ravel(), cHq_t.ravel(), cVq_t.ravel(), cDq_t.ravel()]))
	
	# Reconstruct with thresholded coefficients
	y = pywt.waverec2((cAq_t * qstep, (cHq_t * qstep, cVq_t * qstep, cDq_t * qstep)), wave)
	y = y[:img.shape[0], :img.shape[1]]
	return bits, np.clip(y, 0, 255).astype(np.uint8)

def main():
	img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise ValueError(f"Could not read image: {IMG_PATH}")
	H,W = img.shape
	orig_bits = H*W*8

	# Test different keep ratios
	results = []
	
	for keep_ratio in KEEP_RATIOS:
		print(f"\n=== Testing keep_ratio: {keep_ratio} ===")
		
		# Huffman (no thresholding applied)
		if keep_ratio == 1.0:  # Only run once for Huffman
			hb, hrecon = do_huffman(img)
			ph = pct_smaller(orig_bits, hb)
			results.append(('Huffman', keep_ratio, hb, ph, hrecon))
		
		# DCT with zigzag + RLE (no thresholding)
		db, drecon = do_dct(img, DCT_QSTEP, keep_ratio)
		pdct = pct_smaller(orig_bits, db)
		results.append(('DCT', keep_ratio, db, pdct, drecon))
		
		# DWT with thresholding
		if HAS_PYWT:
			wb, wrecon = do_dwt(img, WAVELET, DWT_QSTEP, keep_ratio)
			pwdt = pct_smaller(orig_bits, wb)
			results.append(('DWT', keep_ratio, wb, pwdt, wrecon))
		else:
			results.append(('DWT', keep_ratio, 0, 0, None))

	# Print results
	print("\n" + "="*60)
	print("COMPARISON RESULTS")
	print("="*60)
	print(f"{'Method':<10} {'Keep Ratio':<12} {'Bits':<12} {'Compression %':<15}")
	print("-"*60)
	
	for method, keep_ratio, bits, comp_pct, recon in results:
		if recon is not None:
			print(f"{method:<10} {keep_ratio:<12} {bits:<12} {comp_pct:<15.1f}")

	# Plot results
	num_methods = 3 if HAS_PYWT else 2
	cols = len(KEEP_RATIOS) + 1  # +1 for original
	rows = num_methods
	
	plt.figure(figsize=(4*cols, 4*rows))
	
	# Original image
	plt.subplot(rows, cols, 1)
	plt.title("Original")
	plt.imshow(img, cmap='gray')
	plt.axis('off')
	
	# Plot each method and keep ratio
	plot_idx = 2  # Start from second position
	
	for method in ['Huffman', 'DCT', 'DWT']:
		if method == 'Huffman' and not any(r[0] == 'Huffman' for r in results):
			continue
		if method == 'DWT' and not HAS_PYWT:
			continue
			
		for keep_ratio in KEEP_RATIOS:
			# Find result for this method and keep ratio
			result = next((r for r in results if r[0] == method and r[1] == keep_ratio), None)
			if result and result[4] is not None:
				method_name, kr, bits, comp_pct, recon_img = result
				
				plt.subplot(rows, cols, plot_idx)
				if method == 'Huffman':
					plt.title(f"Huffman\n{comp_pct:.1f}% smaller")
				elif method == 'DCT':
					plt.title(f"DCT (Zigzag+RLE)\n{comp_pct:.1f}% smaller")
				else:
					plt.title(f"DWT (keep={kr})\n{comp_pct:.1f}% smaller")
				plt.imshow(recon_img, cmap='gray')
				plt.axis('off')
				
				plot_idx += 1
			else:
				plot_idx += 1  # Skip if no result
	
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()
