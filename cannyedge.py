import cv2
import numpy as np
import matplotlib.pyplot as plt
# Import the dependencies


# Canny edge detection


# Custom convolution functions
def filter2D_custom(img, kernel):
    kh, kw = kernel.shape
    h, w = img.shape
    ph, pw = kh // 2, kw // 2
    output = np.zeros_like(img, dtype=np.float32)
    for i in range(ph, h - ph):
        for j in range(pw, w - pw):
            region = img[i - ph:i + ph + 1, j - pw:j + pw + 1]
            output[i, j] = np.sum(region * kernel)
    return output

def same_convolve(input_img, kernel):
    h1, w1 = input_img.shape
    ph2 = kernel.shape[0] // 2
    pw2 = kernel.shape[1] // 2
    h2 = h1 + 2 * ph2
    w2 = w1 + 2 * pw2
    zero_padded_img = np.zeros((h2, w2))
    zero_padded_img[ph2:ph2 + h1, pw2:pw2 + w1] = input_img
    output_img = filter2D_custom(zero_padded_img, kernel)
    return output_img[ph2:ph2 + h1, pw2:pw2 + w1]

# Custom Canny implementation
def custom_canny(image, T1, T2):
    g_kernel = np.array([[1/16, 2/16, 1/16],
                         [2/16, 4/16, 2/16],
                         [1/16, 2/16, 1/16]], dtype=np.float32)
    sigma = 4               # standard deviation of Gaussian
    k = int(np.ceil(6 * sigma))
    
    if k % 2 == 0:
       k += 1 
    blurred = cv2.GaussianBlur(image, (k, k), sigma)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
    gx = same_convolve(blurred, sobel_x)
    gy = same_convolve(blurred, sobel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180
    '''if (-22.5 <= angle[i, j] < 22.5) or (-157.5 < angle[i, j] <= 157.5):
                horizontal edge
            elif (22.5 <= angle[i, j] < 67.5) or  (-112.5 < angle[i, j] <= -157.5):
               -45 degree or left diagonal edge
            elif (67.5 <= angle[i, j] < 112.5) or(- 67.5 < angle[i, j] <= -112.5):
                +45 degree or right diagoal edge
            elif (112.5 <= angle[i, j] < 157.5)or(-22.5 < angle[i, j] <= -67.5) :
                vertical edge
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0 '''
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0
    strong = 255
    weak = 75
    result = np.zeros_like(Z, dtype=np.uint8)
    strong_i, strong_j = np.where(Z >= T2)
    weak_i, weak_j = np.where((Z >= T1) & (Z < T2))
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if result[i, j] == weak:
                if ((result[i + 1, j - 1] == strong) or (result[i + 1, j] == strong) or (result[i + 1, j + 1] == strong)
                    or (result[i, j - 1] == strong) or (result[i, j + 1] == strong)
                    or (result[i - 1, j - 1] == strong) or (result[i - 1, j] == strong) or (result[i - 1, j + 1] == strong)):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result

# Main function to run
def main():
    image = cv2.imread('Fig1022(a)(building_original).tif',0)
    scale=1/5
    if image is None:
        print("Error: Image not found.")
        return
    TL = 0.04               # low threshold (normalized)
    TH = 0.10               # high threshold (normalized)
    # smallest int >= 24
            

    # Convert MATLAB thresholds (0–1) to OpenCV scale (0–255)
    T1 = int(TL * 255)
    T2 = int(TH * 255)
    height, width = image.shape        
    #heightScale = int(height * scale)
    #widthScale = int(width * scale)
    #image = cv2.resize(image, (widthScale, heightScale), interpolation=cv2.INTER_LINEAR)

    sigmas = [4]     # 0 = no blur
    kernel_sizes = [7,11,15,19,25]


    for k in kernel_sizes:
        #print(f"sigma={sigma}, kernel={k}x{k}")
        img_blur = cv2.GaussianBlur(image, (k, k), 4)

        edges = custom_canny(img_blur, T1,T2)
        cvedge=cv2.Canny(img_blur, T1,T2)
        cv2.imwrite(f"Canny(kernel={k}).jpg", edges)
        cv2.imwrite(f"cvCanny(kernel={k}).jpg", cvedge)
  
  
  

   

   

if __name__ == "__main__":
    main()
