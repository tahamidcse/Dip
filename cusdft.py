import cv2
import numpy as np
import matplotlib.pyplot as plt
def fftshift_2d(x):
    M, N = x.shape
    m_mid = M // 2
    n_mid = N // 2
    # Rearrange 4 quadrants
    return np.block([
        [x[m_mid:, n_mid:], x[m_mid:, :n_mid]],
        [x[:m_mid, n_mid:], x[:m_mid, :n_mid]]
    ])

# --- Manual 1D DFT and IDFT ---
def _DFT(x, N):
    result = np.zeros(N, dtype=np.complex128)
    for m in range(N):
        for n in range(N):
            result[m] += x[n] * np.exp(-1j * 2 * np.pi * m * n / N)
    return result

def _IDFT(dft, N):
    result = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        for m in range(N):
            result[n] += dft[m] * np.exp(1j * 2 * np.pi * m * n / N)
    result /= N
    return result


# --- 2D DFT and IDFT using row- and column-wise 1D DFT ---
def DFT2D(img):
    M, N = img.shape
    temp = np.zeros((M, N), dtype=np.complex128)
    result = np.zeros((M, N), dtype=np.complex128)

    # Apply 1D DFT to rows
    for i in range(M):
        temp[i, :] = _DFT(img[i, :], N)
    
    # Apply 1D DFT to columns
    for j in range(N):
        result[:, j] = _DFT(temp[:, j], M)

    return result

def IDFT2D(dft2d):
    M, N = dft2d.shape
    temp = np.zeros((M, N), dtype=np.complex128)
    result = np.zeros((M, N), dtype=np.complex128)

    # Apply 1D IDFT to columns
    for j in range(N):
        temp[:, j] = _IDFT(dft2d[:, j], M)
    
    # Apply 1D IDFT to rows
    for i in range(M):
        result[i, :] = _IDFT(temp[i, :], N)

    return np.real(result)


# --- Load grayscale image using OpenCV ---
img = cv2.imread("/content/dft.jpeg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32)).astype(np.float32)

# --- Manual 2D DFT ---
manual_dft = DFT2D(img)
manual_idft=IDFT2D(manual_dft)
manual_magnitude =np.log1p( np.abs(fftshift_2d(manual_dft)))
manual_phase = np.angle(manual_dft)

manual_reconstructed = IDFT2D(manual_dft)
dft=np.fft.fft2(img)
shifted_dft = np.fft.fftshift(dft)
magnitude = np.log1p(np.abs(shifted_dft))
diff=np.abs(manual_magnitude-magnitude)


# --- Visualization ---
plt.figure(figsize=(12, 10))

plt.subplot(3,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image in float32")

plt.subplot(3,2,2)
plt.imshow(manual_magnitude, cmap='hot')
plt.title("Manual DFT Magnitude (log scale)")

plt.subplot(3,2,3)
plt.imshow(magnitude, cmap='hot')
plt.title("OpenCV DFT Magnitude (log scale)")



plt.subplot(3,2,4)
plt.imshow(np.clip(manual_idft, 0, 255).astype(np.uint8), cmap='gray')
plt.title("Reconstructed Image (IDFT)")
plt.axis('off')


plt.tight_layout()
plt.show()
