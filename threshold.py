# Python program to illustrate simple thresholding on an image

# Importing libraries
import cv2 
import numpy as np 
import matplotlib.pyplot as plt

# Function to compute histogram
def hist(img):
    pixel_count = np.zeros((256,), dtype=np.uint)

    # Count number of pixels
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            pixel_count[pixel_value] += 1

    print(pixel_count)
    return pixel_count

# Read image
image1 = cv2.imread('image1.jpeg')

# Convert the image to grayscale 
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Create copies for thresholding
img2 = img.copy()
img3 = img.copy()

# Get image dimensions
height, width = img.shape

print("Height:", height)
print("Width:", width)

# First thresholding
for i in range(height):
    for j in range(width):
        r = img[i][j]
        if 120 < r < 140:
            r = 255 
        img[i][j] = r

# Second thresholding (binary mask)
for i in range(height):
    for j in range(width):
        r = img2[i][j]
        if 120 < r < 140:
            r = 255 
        else:
            r = 0    
        img2[i][j] = r

# Third thresholding (multi-level)
for i in range(height):
    for j in range(width):
        r = img3[i][j]
        if 120 < r < 140:
            r = 127
        elif 140 < r < 160:
            r = 255
        else:
            r = 0
        img3[i][j] = r    

# Display the processed image using matplotlib
x = np.arange(256)
pixel_count1 = hist(img)
pixel_count2 = hist(img2)
pixel_count3 = hist(img3)

plt.figure()#figsize=(30, 20))

plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Thresholded Image 1')



plt.subplot(3, 2, 2)
plt.bar(x, pixel_count1)
plt.title('Bar Histogram 1')

plt.subplot(3, 2, 3)
plt.imshow(img2, cmap='gray')
plt.title('Thresholded Image 2')


plt.subplot(3, 2, 4)
plt.bar(x, pixel_count2)
plt.title('Bar Histogram 2')

plt.subplot(3, 2, 5)
plt.imshow(img3, cmap='gray')
plt.title('Thresholded Image 3')


plt.subplot(3, 2, 6)
plt.bar(x, pixel_count3)
plt.title('Bar Histogram 3')

plt.tight_layout()
plt.show()
plt.close()
