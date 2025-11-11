# Using the PyWavelets module, available at:
# https://pywavelets.readthedocs.io/en/latest/install.html

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Set figure size and font size
plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

# --------------------------------------------
# Load image using OpenCV and convert to grayscale
# --------------------------------------------

# Replace the path with the correct image location
img_path = 'dwt.jpeg'  # Update if needed
A = cv2.imread(img_path)

if A is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# Convert BGR (OpenCV default) to grayscale
B = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
B = B.astype(np.float32) / 255.0  # Normalize for better results

# --------------------------------------------
# Wavelet decomposition (2 levels)
# --------------------------------------------

n = 2              # Number of decomposition levels
w = 'haar'          # Wavelet type


#coeffs=pywt.dwt2(B,'haar')

# Normalize each coefficient array

coeffs= pywt.wavedec2(B, wavelet=w, level=n)


# Normalize approximation + each detail level

coeffs[0] = coeffs[0] / np.abs(coeffs[0]).max()
for i in range(1, len(coeffs)):
    coeffs[i] = tuple(d / np.abs(d).max() for d in coeffs[i])


# Convert to array for visualization
arr, coeff_slices = pywt.coeffs_to_array(coeffs)

plt.imshow(arr, cmap='gray_r', vmin=-0.25, vmax=0.75)
plt.title('Wavelet Coefficients (2-level DWT')
plt.axis('off')
plt.savefig("output.png", bbox_inches='tight', pad_inches=0)
plt.show()

    

