import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 2D FFT Shift (custom)
# -----------------------------
def fftshift_2d(x):
    M, N = x.shape
    m_mid = M // 2
    n_mid = N // 2
    return np.block([
        [x[m_mid:, n_mid:], x[m_mid:, :n_mid]],
        [x[:m_mid, n_mid:], x[:m_mid, :n_mid]]
    ])

# -----------------------------
# Memoized 1D FFT (DP approach)
# -----------------------------
fft_cache = {}

def _DFT(x):
    N = len(x)
    key = tuple(np.round(x, 8))  # hashable key

    if key in fft_cache:
        return fft_cache[key]

    if N <= 1:
        fft_cache[key] = x
        return x

    # Divide
    even = _DFT(x[::2])
    odd  = _DFT(x[1::2])

    # Twiddle factors
    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    X = np.zeros(N, dtype=complex)
    half = N // 2
    for k in range(half):
        X[k] = even[k] + factor[k] * odd[k]
        X[k + half] = even[k] - factor[k] * odd[k]

    fft_cache[key] = X
    return X

# -----------------------------
# Inverse FFT using DP (IFFT)
# -----------------------------
def _IDFT(X):
    N = len(X)
    if N <= 1:
        return X
    return np.conjugate(_DFT(np.conjugate(X))) / N

# -----------------------------
# 2D FFT / IFFT via row & column transform
# -----------------------------
def DFT2D(img):
    M, N = img.shape
    temp = np.zeros((M, N), dtype=np.complex128)
    result = np.zeros((M, N), dtype=np.complex128)

    for i in range(M):
        temp[i, :] = _DFT(img[i, :])

    for j in range(N):
        result[:, j] = _DFT(temp[:, j])

    return result

def IDFT2D(dft2d):
    M, N = dft2d.shape
    temp = np.zeros((M, N), dtype=np.complex128)
    result = np.zeros((M, N), dtype=np.complex128)

    for j in range(N):
        temp[:, j] = _IDFT(dft2d[:, j])
    for i in range(M):
        result[i, :] = _IDFT(temp[i, :])

    return np.real(result)

# -----------------------------
# Load and process image
# -----------------------------
img = cv2.imread("/content/dft.jpeg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32)).astype(np.float32)

manual_dft = DFT2D(img)
manual_magnitude = np.log1p(np.abs(fftshift_2d(manual_dft)))
manual_reconstructed = IDFT2D(manual_dft)

# Reference using NumPy
dft = np.fft.fft2(img)
shifted_dft = np.fft.fftshift(dft)
magnitude = np.log1p(np.abs(shifted_dft))

diff = np.abs(manual_magnitude - magnitude)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12, 10))

plt.subplot(3,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")

plt.subplot(3,2,2)
plt.imshow(manual_magnitude, cmap='hot')
plt.title("Manual FFT (DP Memoized) Magnitude")

plt.subplot(3,2,3)
plt.imshow(magnitude, cmap='hot')
plt.title("NumPy FFT Magnitude")

plt.subplot(3,2,4)
plt.imshow(np.clip(manual_reconstructed, 0, 255).astype(np.uint8), cmap='gray')
plt.title("Reconstructed Image (IDFT)")

plt.subplot(3,2,5)
plt.imshow(diff, cmap='gray')
plt.title("Difference between Manual & NumPy FFT")

plt.tight_layout()
plt.show()
