import cv2, numpy as np, heapq
from collections import Counter
import matplotlib.pyplot as plt
import pywt



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
def zigzag_scan(matrix):
    """Zigzag scan of a 2D matrix"""
    rows, cols = matrix.shape
    result = []
    for s in range(rows + cols - 1):
        if s % 2 == 0:
            for i in range(s, -1, -1):
                j = s - i
                if i < rows and j < cols:
                    result.append(matrix[i, j])
        else:
            for j in range(s, -1, -1):
                i = s - j
                if i < rows and j < cols:
                    result.append(matrix[i, j])
    return np.array(result)


def huff_bits(arr: np.ndarray) -> int:
    flat = arr.ravel().tolist()
    if not flat:
        return 0
    freq = Counter(flat)
    h = []
    uid = 0
    for s, f in freq.items():
        heapq.heappush(h, (f, uid, (s, None, None)))
        uid += 1
    while len(h) > 1:
        f1, _, n1 = heapq.heappop(h)
        f2, _, n2 = heapq.heappop(h)
        heapq.heappush(h, (f1 + f2, uid, (None, n1, n2)))
        uid += 1
    root = h[0][2]
    codes = {}

    def walk(node, p=""):
        s, l, r = node
        if s is not None:
            codes[s] = p or "0"
            return
        walk(l, p + "0")
        walk(r, p + "1")

    walk(root)
    return sum(len(codes[v]) for v in flat)


def pct_smaller(orig_bits, comp_bits):
    return max(0.0, 100.0 * (1.0 - comp_bits / max(1, orig_bits)))




'''
def apply_thresholding(coeffs, keep_ratio):
    """Apply thresholding to DWT coefficients based on keep ratio"""
    if keep_ratio >= 1.0:
        return coeffs  # No thresholding

    coeff_flat = []
    coeff_flat.append(coeffs[0].ravel())
    for detail_coeff in coeffs[1]:
        coeff_flat.append(detail_coeff.ravel())
    all_coeffs = np.concatenate(coeff_flat)

    abs_coeffs = np.abs(all_coeffs)
    sorted_indices = np.argsort(abs_coeffs)
    thresh_idx = int(len(abs_coeffs) * (1 - keep_ratio))
    threshold = abs_coeffs[sorted_indices[thresh_idx]] if thresh_idx > 0 else 0

    start_idx = 0
    new_cA = coeffs[0].copy()
    new_details = []

    cA_size = coeffs[0].size
    cA_flat = all_coeffs[start_idx:start_idx + cA_size]
    cA_flat[np.abs(cA_flat) < threshold] = 0
    new_cA = cA_flat.reshape(coeffs[0].shape)
    start_idx += cA_size

    for i, detail_coeff in enumerate(coeffs[1]):
        detail_size = detail_coeff.size
        detail_flat = all_coeffs[start_idx:start_idx + detail_size]
        detail_flat[np.abs(detail_flat) < threshold] = 0
        new_details.append(detail_flat.reshape(detail_coeff.shape))
        start_idx += detail_size

    return (new_cA, tuple(new_details))
 '''

# ====================== Compression Methods ===============================

def do_huffman(img):
    bits = huff_bits(img)
    return bits, img.copy()


def do_dct(img, qstep=10.0, keep_ratio=1.0):
    x = img.astype(np.float32) - 128.0
    C = cv2.dct(x)
    Cq = np.rint(C / qstep).astype(np.int32)

    zigzag_coeffs = zigzag_scan(Cq)
    rle_pairs = run_length_encode(zigzag_coeffs)

    flat_rle = []
    for value, run in rle_pairs:
        flat_rle.extend([value, run])
    bits = huff_bits(np.array(flat_rle)) if flat_rle else 0

    y = cv2.idct(Cq.astype(np.float32) * qstep) + 128.0
    return bits, np.clip(y, 0, 255).astype(np.uint8)

'''
def do_dwt(img, wave="haar", qstep=1.0, keep_ratio=1.0):
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
    if keep_ratio < 1.0:
        coeffs_q = apply_thresholding(coeffs_q, keep_ratio)

    cAq_t, (cHq_t, cVq_t, cDq_t) = coeffs_q
    bits = huff_bits(np.concatenate([
        cAq_t.ravel(), cHq_t.ravel(), cVq_t.ravel(), cDq_t.ravel()
    ]))

    y = pywt.waverec2((cAq_t * qstep, (cHq_t * qstep, cVq_t * qstep, cDq_t * qstep)), wave)
    y = y[:img.shape[0], :img.shape[1]]
    return bits, np.clip(y, 0, 255).astype(np.uint8)
'''
def main():
    IMG_PATH = "dwtdct.jpg"
    DCT_QSTEP = 10.0
    
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {IMG_PATH}")

    H, W = img.shape
    orig_bits = H * W * 8
    results = []

    print("\n=== Testing Compression Methods ===")

    # Huffman compression
    hb, hrecon = do_huffman(img)
    ph = pct_smaller(orig_bits, hb)
    results.append(('Huffman', hb, ph, hrecon))

    # DCT with RLE + Huffman compression
    db, drecon = do_dct(img, DCT_QSTEP)
    pdct = pct_smaller(orig_bits, db)
    results.append(('DCT', db, pdct, drecon))

    # Print results
    print("\n" + "=" * 50)
    print("COMPRESSION RESULTS")
    print("=" * 50)
    print(f"{'Method':<10} {'Bits':<12} {'Compression %':<15}")
    print("-" * 50)
    for method, bits, comp_pct, _ in results:
        print(f"{method:<10} {bits:<12} {comp_pct:<15.1f}")

    # Calculate bits per pixel
    total_pixels = H * W
    print("\n" + "=" * 50)
    print("BITS PER PIXEL (BPP) ANALYSIS")
    print("=" * 50)
    print(f"Original: {8:.2f} bpp")
    for method, bits, comp_pct, _ in results:
        bpp = bits / total_pixels
        print(f"{method}: {bpp:.2f} bpp")

    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # Huffman compressed
    plt.subplot(1, 3, 2)
    h_method, h_bits, h_comp, h_recon = results[0]
    plt.title(f"Huffman Compression\n{h_comp:.1f}% smaller\n{h_bits} bits")
    plt.imshow(h_recon, cmap='gray')
    plt.axis('off')

    # DCT compressed
    plt.subplot(1, 3, 3)
    d_method, d_bits, d_comp, d_recon = results[1]
    plt.title(f"DCT + RLE + Huffman\n{d_comp:.1f}% smaller\n{d_bits} bits")
    plt.imshow(d_recon, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
