import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Manual 1D DCT and IDCT ---
def _DCT_1D(x):
    N = len(x)
    result = np.zeros(N, dtype=np.float64)
    for k in range(N):
        alpha = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
        for n in range(N):
            result[k] += x[n] * np.cos(np.pi * (2*n + 1) * k / (2*N))
        result[k] *= alpha
    return result

def _IDCT_1D(X):
    N = len(X)
    result = np.zeros(N, dtype=np.float64)
    for n in range(N):
        for k in range(N):
            alpha = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
            result[n] += alpha * X[k] * np.cos(np.pi * (2*n + 1) * k / (2*N))
    return result


# --- 2D DCT and IDCT using row- and column-wise 1D DCT ---
def DCT2D(img):
    M, N = img.shape
    temp = np.zeros((M, N), dtype=np.float64)
    result = np.zeros((M, N), dtype=np.float64)

    # Apply 1D DCT to rows
    for i in range(M):
        temp[i, :] = _DCT_1D(img[i, :])

    # Apply 1D DCT to columns
    for j in range(N):
        result[:, j] = _DCT_1D(temp[:, j])

    return result

def IDCT2D(dct):
    M, N = dct.shape
    temp = np.zeros((M, N), dtype=np.float64)
    result = np.zeros((M, N), dtype=np.float64)

    # Apply 1D IDCT to columns
    for j in range(N):
        temp[:, j] = _IDCT_1D(dct[:, j])

    # Apply 1D IDCT to rows
    for i in range(M):
        result[i, :] = _IDCT_1D(temp[i, :])

    return result


# --- Load grayscale image ---
img = cv2.imread("/content/dft.jpeg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32)).astype(np.float64)

# --- Manual 2D DCT and IDCT ---
manual_dct = DCT2D(img)
manual_idct = IDCT2D(manual_dct)

# --- OpenCV built-in DCT for comparison ---
opencv_dct = cv2.dct(img)
opencv_idct = cv2.idct(opencv_dct)

# --- Difference between manual and OpenCV DCT ---
diff = np.abs(manual_dct - opencv_dct)

# --- Visualization ---
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image (float64)")

plt.subplot(3, 2, 2)
plt.imshow(np.clip(manual_idct, 0, 255).astype(np.uint8), cmap='gray')
plt.title("Reconstructed (Manual IDCT)")



plt.subplot(3, 2, 3)
plt.imshow(np.log1p(np.abs(opencv_dct)), cmap='hot')
plt.title("OpenCV 2D DCT (log scale)")

plt.subplot(3, 2, 4)
plt.imshow(np.log1p(np.abs(manual_dct)), cmap='hot')
plt.title("Manual 2D DCT (log scale)")
plt.axis('off')

plt.tight_layout()
plt.show()
