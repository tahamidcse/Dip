# 1st part link https://github.com/tahamidcse/Dip/blob/main/he_11.py
#2nd part link https://github.com/tahamidcse/Dip/blob/main/local_transformations.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt


import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# Common Histogram Utilities
# ------------------------------------------
def compute_histogram(tile):
    return np.bincount(tile.ravel(), minlength=256).astype(np.float32)

def compute_cdf(hist):
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    cdf_normalized = ((cdf - cdf_min) * 255) / (cdf[-1] - cdf_min)
    return cdf_normalized.astype(np.uint8)

# ------------------------------------------
# AHE - Tile equalization without clipping
# ------------------------------------------
def equalize_tile(tile):
    hist = compute_histogram(tile)
    return compute_cdf(hist)

# ------------------------------------------
# CLAHE - Tile equalization with clipping
# ------------------------------------------
def compute_clipped_histogram(tile, clip_limit):
    hist = compute_histogram(tile)
    excess = hist - clip_limit
    excess[excess < 0] = 0
    total_excess = int(excess.sum())
    hist = np.minimum(hist, clip_limit)

    # Redistribute excess
    bin_incr = total_excess // 256
    remainder = total_excess % 256
    hist += bin_incr
    hist[:remainder] += 1
    return hist

def equalize_tile_clahe(tile, clip_limit):
    hist = compute_clipped_histogram(tile, clip_limit)
    return compute_cdf(hist)

# ------------------------------------------
# Core Function for Adaptive HE or CLAHE
# ------------------------------------------
def adaptive_hist_equalization_general(img, tile_size=(8, 8), tile_equalizer=None):
    h, w = img.shape
    tile_h, tile_w = tile_size
    n_tiles_y = (h + tile_h - 1) // tile_h
    n_tiles_x = (w + tile_w - 1) // tile_w

    pad_h = n_tiles_y * tile_h - h
    pad_w = n_tiles_x * tile_w - w
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    H, W = img_padded.shape

    mappings = np.zeros((n_tiles_y, n_tiles_x, 256), dtype=np.uint8)

    # Compute per-tile equalization mappings
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            y1, y2 = i * tile_h, (i + 1) * tile_h
            x1, x2 = j * tile_w, (j + 1) * tile_w
            tile = img_padded[y1:y2, x1:x2]
            mappings[i, j] = tile_equalizer(tile)#normalized cdf

    # Interpolate mappings across the image
    output = np.zeros_like(img_padded)
    for intensity in range(256):
        map_layer = mappings[:, :, intensity]
        interp_map = cv2.resize(map_layer, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = img_padded == intensity
        output[mask] = interp_map[mask]

    return output[:h, :w]


img=cv2.imread('Fig0326(a)(embedded_square_noisy_512).tif',0)

ahe_img = adaptive_hist_equalization_general(
    img,
    tile_size=(12, 12),
    tile_equalizer=equalize_tile
)

clip_limit = 40.0
ahe_clahe_img = adaptive_hist_equalization_general(
    img,
    tile_size=(12, 12),
    tile_equalizer=lambda tile: equalize_tile_clahe(tile, clip_limit)
)

clahe_cv = cv2.createCLAHE(clip_limit, tileGridSize=(12, 12)).apply(img)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(ahe_img, cmap='gray')
axes[1].set_title("Custom AHE")
axes[1].axis('off')
cv2.imwrite
axes[2].imshow(ahe_clahe_img, cmap='gray')
axes[2].set_title("Custom CLAHE")
axes[2].axis('off')

axes[3].imshow(clahe_cv, cmap='gray')
axes[3].set_title("OpenCV CLAHE")
axes[3].axis('off')

plt.tight_layout()
plt.show()

