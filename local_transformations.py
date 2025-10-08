

import numpy as np
import cv2
import matplotlib.pyplot as plt

def neg(img_gray):
    return 255 - img_gray

def gamma(img_gray, gamma_value=0.5):
    # Normalize to 0-1
    img = img_gray / 255.0
    img_gamma = img ** gamma_value
    img_gamma = np.uint8(img_gamma * 255)
    return img_gamma

def apply_tile_operations(img, tile_size=(8, 8), operation_map=None):
    """
    Apply different operations to different tiles in the image.

    Parameters:
        img (2D np.array): Grayscale input image
        tile_size (tuple): Height and width of each tile
        operation_map (function): A function that takes tile indices (i, j)
                                  and returns an operation function for that tile

    Returns:
        Processed image
    """
    h, w = img.shape
    tile_h, tile_w = tile_size

    n_tiles_y = (h + tile_h - 1) // tile_h
    n_tiles_x = (w + tile_w - 1) // tile_w

    # Pad image if necessary
    pad_h = n_tiles_y * tile_h - h
    pad_w = n_tiles_x * tile_w - w
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

    H, W = img_padded.shape
    result_img = np.zeros_like(img_padded)

    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            y1, y2 = i * tile_h, (i + 1) * tile_h
            x1, x2 = j * tile_w, (j + 1) * tile_w
            tile = img_padded[y1:y2, x1:x2]

            # Get operation for this tile
            op = operation_map(i, j)
            processed_tile = op(tile)

            result_img[y1:y2, x1:x2] = processed_tile

    # Crop to original size
    return result_img[:h, :w]

def example_operation_map(i, j):
    # Alternate pattern: neg on even tiles, gamma on odd
    if (i + j) % 2 == 0:
        return neg
    else:
        return lambda img: gamma(img, gamma_value=0.4)  # adjust gamma as needed

# Load grayscale image
img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply tile-based operations
processed_img = apply_tile_operations(img, tile_size=(64, 64), operation_map=example_operation_map)

# Show the result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed_img, cmap='gray')
plt.title('Tile-Based Processed Image')
plt.axis('off')

plt.tight_layout()
plt.show()
