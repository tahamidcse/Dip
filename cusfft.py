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
def _DFT(x,N):
    N = len(x)
    # Bit-reversal permutation
    bits = int(np.log2(N))
    indices = np.arange(N)
    reversed_indices = np.array([int(f'{i:0{bits}b}'[::-1], 2) for i in indices])
    x = np.array(x, dtype=complex)[reversed_indices]

    # Iterative butterfly computation
    half_size = 1
    while half_size < N:
        step = half_size * 2
        exp_factor = np.exp(-2j * np.pi * np.arange(half_size) / step)
        for k in range(0, N, step):
            for n in range(half_size):
                even = x[k + n]
                odd = exp_factor[n] * x[k + n + half_size]
                x[k + n] = even + odd
                x[k + n + half_size] = even - odd
        half_size = step
    return x

def _IDFT(X, N):
    N = len(X)
    X_conj = np.conjugate(X)
    x_time = _DFT(X_conj,len(X_conj))
    return np.conjugate(x_time) / N

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
plt.imshow(np.clip(manual_idft, 0, 255).astype(np.uint8), cmap='gray')
plt.title("Reconstructed Image (IDFT)")


plt.subplot(3,2,3)
plt.imshow(magnitude, cmap='hot')
plt.title("OpenCV DFT Magnitude (log scale)")



plt.subplot(3,2,4)
plt.imshow(manual_magnitude, cmap='hot')
plt.title("Manual DFT Magnitude (log scale)")
plt.axis('off')


plt.tight_layout()
plt.show()
