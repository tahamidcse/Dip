import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1D DCT (Type-II)
# ----------------------------------------
def DCT_1D(x, N):
    X = np.zeros(N)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.cos(np.pi * (n + 0.5) * k / N)
    return X

# ----------------------------------------
# 1D Inverse DCT (Type-III)
# ----------------------------------------
def IDCT_1D(X, N):
    x = np.zeros(N)
    for n in range(N):
        x[n] = X[0] / 2
        for k in range(1, N):
            x[n] += X[k] * np.cos(np.pi * (n + 0.5) * k / N)
    x *= 2 / N
    return x

# ----------------------------------------
# 2D DCT
# ----------------------------------------
def DCT_2D(img):
    M, N = img.shape
    temp = np.zeros((M, N))
    result = np.zeros((M, N))

    # DCT on rows
    for i in range(M):
        temp[i, :] = DCT_1D(img[i, :], N)

    # DCT on columns
    for j in range(N):
        result[:, j] = DCT_1D(temp[:, j], M)

    return result

# ----------------------------------------
# 2D IDCT
# ----------------------------------------
def IDCT_2D(dct_img):
    M, N = dct_img.shape
    temp = np.zeros((M, N))
    result = np.zeros((M, N))

    # IDCT on columns
    for j in range(N):
        temp[:, j] = IDCT_1D(dct_img[:, j], M)

    # IDCT on rows
    for i in range(M):
        result[i, :] = IDCT_1D(temp[i, :], N)

    return result

# ----------------------------------------
# Custom DCT shift (centerize low frequencies)
# ----------------------------------------
def dctshift(dct_img):
    """Shift DCT quadrants (similar to np.fft.fftshift)."""
    M, N = dct_img.shape
    M2, N2 = M // 2, N // 2
    shifted = np.zeros_like(dct_img)

    # swap quadrants
    shifted[:M2, :N2] = dct_img[M2:, N2:]
    shifted[M2:, N2:] = dct_img[:M2, :N2]
    shifted[:M2, N2:] = dct_img[M2:, :N2]
    shifted[M2:, :N2] = dct_img[:M2, N2:]
    return shifted

# ----------------------------------------
# Load grayscale image
# ----------------------------------------
img_path = 'dft.jpeg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
A = cv2.resize(img, (32, 32))

if A is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

A = A.astype(np.float32) / 255.0

# ----------------------------------------
# Apply custom 2D DCT & IDCT
# ----------------------------------------
dct_custom = DCT_2D(A)
dct_centered = np.fft.fftshift(dct_custom)  # Centerize DCT coefficients
idct_custom = IDCT_2D(dct_custom)

# ----------------------------------------
# Apply OpenCV DCT & IDCT
# ----------------------------------------
dct_cv2 = cv2.dct(A)
dctcv_centered=np.fft.fftshift(dct_cv2)  
idct_cv2 = cv2.idct(dct_cv2)

# ----------------------------------------
# Compute errors
# ----------------------------------------
dct_error = np.abs(dct_custom - dct_cv2).mean()
idct_error = np.abs(idct_custom - idct_cv2).mean()
recon_error = np.abs(A - idct_custom).mean()

print(f"Mean |Custom DCT - cv2.DCT| error = {dct_error:.6e}")
print(f"Mean |Custom IDCT - cv2.IDCT| error = {idct_error:.6e}")
print(f"Mean |Original - Custom IDCT| error = {recon_error:.6e}")

# ----------------------------------------
# Display comparison
# ----------------------------------------
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(A, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(np.log1p(np.abs(dctcv_centered)), cmap='hot')
plt.title("dctcv2_centered")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(np.log1p(np.abs(dct_centered)), cmap='hot')
plt.title("DCT custom(Centered / Shifted)")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(np.clip(idct_custom, 0, 1), cmap='gray')
plt.title("Custom IDCT Reconstruction")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(np.clip(idct_cv2, 0, 1), cmap='gray')
plt.title("OpenCV IDCT Reconstruction")
plt.axis('off')



plt.tight_layout()
plt.show()
