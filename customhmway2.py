from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    # Load grayscale images
    src_path = "Fig0309(a)(washed_out_aerial_image).tif"
    ref_path = "Fig0316(3).tif"
    
    src_img = cv2.imread(src_path, 0)  # Read as grayscale
    ref_img = cv2.imread(ref_path, 0)

    if src_img is None or ref_img is None:
        print("Error loading images.")
        return

    # 1. Equalize both images and get mappings
    eq_src_img, eq_src_map = equalize_histogram(src_img)
    eq_ref_img, eq_ref_map = equalize_histogram(ref_img)

    # 2. Invert reference equalization mapping
    inv_ref_map = invert_mapping(eq_ref_map)

    # 3. Build transformation: map equalized source to inverse of reference
    final_map = build_transform_map(eq_src_map, inv_ref_map)

    # 4. Apply the final transformation to the original source image
    matched_img = img_conv(src_img, final_map)

    # 5. Match the histogram using skimage
    matched_builtin = match_histograms(src_img, ref_img, channel_axis=None)
    matched_builtin_uint8 = np.clip(matched_builtin, 0, 255).astype(np.uint8)

    # 6. Plot histograms for comparison
    plot_histogram_comparison(src_img, matched_img, matched_builtin_uint8)

    # 7. Display original, reference, matched images and their histograms
    display_results(src_img, ref_img, matched_img, matched_builtin_uint8)

#================= Histogram Calculation ==============================
def histogram(img_2D):
    h, w = img_2D.shape
    hist = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            intensity = int(img_2D[i, j])  # Ensure it's int
            hist[intensity] += 1
    return hist

#================= PDF and CDF ========================================
def pdf_f(hist):
    return hist / hist.sum()

def cdf_f(pdf):
    return np.cumsum(pdf)

#================= Histogram Equalization =============================
def equalize_histogram(img):
    hist = histogram(img)
    pdf = pdf_f(hist)
    cdf = cdf_f(pdf)
    equalized_map = (cdf * 255).astype(np.uint8)
    equalized_img = equalized_map[img]
    return equalized_img, equalized_map

#================= Invert Mapping =====================================
def invert_mapping(mapping):
    inverse = np.full(256, -1, dtype=int)

    for i in range(256):
        val = mapping[i]
        inverse[val] = i

    last_valid = 0
    for i in range(256):
        if inverse[i] == -1:
            inverse[i] = last_valid
        else:
            last_valid = inverse[i]

    return inverse.astype(np.uint8)

#================= Build Final Mapping ================================
def build_transform_map(eq_src_map, inv_ref_map):
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = inv_ref_map[eq_src_map[i]]
    return mapping

#================= Apply Mapping to Image =============================
def img_conv(img_gray, mapping):
    return mapping[img_gray]

#================= Prepare Histograms for Plotting ====================
def prepare_histogram(img):
    hist = histogram(img)
    pdf = pdf_f(hist)
    return hist, pdf

#================= Plot Histograms for Comparison =====================
def plot_histogram_comparison(original_img, custom_img, skimage_img):
    _, pdf_original = prepare_histogram(original_img)
    _, pdf_custom = prepare_histogram(custom_img)
    _, pdf_builtin = prepare_histogram(skimage_img)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.bar(range(256), pdf_original, width=1.0, color='blue')
    plt.title('Original Image Histogram')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.bar(range(256), pdf_custom, width=1.0, color='green')
    plt.title('Custom Matched Histogram')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.bar(range(256), pdf_builtin, width=1.0, color='red')
    plt.title('Skimage Matched Histogram')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

#================= Display All Results ================================
def display_results(src_img, ref_img, matched_img, matched_builtin):
    # Histograms and CDFs
    src_hist = histogram(src_img)
    ref_hist = histogram(ref_img)
    matched_hist = histogram(matched_img)

    src_cdf = cdf_f(pdf_f(src_hist))
    ref_cdf = cdf_f(pdf_f(ref_hist))

    img_set = [
        src_img, ref_img, matched_img,
        src_hist, ref_hist,
        src_cdf, ref_cdf,
        matched_hist
    ]

    titles = [
        "Source Image", "Reference Image", "Custom Matched Image",
        "Source Histogram", "Reference Histogram",
        "Source CDF", "Reference CDF", "Custom Matched Histogram"
    ]

    plt.figure(figsize=(14, 10))
    for i in range(len(img_set)):
        plt.subplot(3, 3, i + 1)
        if img_set[i].ndim == 2:
            plt.imshow(img_set[i], cmap="gray")
            plt.axis('off')
        else:
            plt.bar(range(256), img_set[i], width=1.0)
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()

    # Display skimage matched image
    plt.figure(figsize=(5, 5))
    plt.imshow(matched_builtin, cmap='gray')
    plt.title('Skimage Matched Image')
    plt.axis('off')
    plt.show()

#================= Run Script =========================================
if __name__ == "__main__":
    main()
