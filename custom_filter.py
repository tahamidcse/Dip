import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def filter2D_custom(input_img, kernel):
    input_h, input_w = input_img.shape
    kernel_h, kernel_w = kernel.shape

    # Flip the kernel to perform convolution instead of correlation
    kernel_flipped = np.flipud(np.fliplr(kernel))

    # Output dimensions (valid convolution)
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1

    output_img = np.zeros((output_h, output_w), dtype=np.float32)

    # Perform convolution
    for h in range(output_h):
        for w in range(output_w):
            roi = input_img[h:h + kernel_h, w:w + kernel_w]
            output_img[h, w] = np.sum(roi * kernel_flipped)

    # Clip and convert to uint8 like cv2.filter2D would do
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    
    return output_img



def main(image_path):
    # Image paths
    imgpath="Fig0338(a)(blurry_moon).tif"
    imgpath2="Fig1022(a)(building_original).tif"
    imgp="doll.jpg"
    
    # Load grayscale images
    imga = cv2.imread(imgp, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imgpath2, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not read the image from {image_path}")
        return

    # Kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32)
    
    laplace = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]], dtype=np.float32)

    averaging = (1/9)*np.ones((3, 3), dtype=np.float32)

    # Apply OpenCV filter2D
    cv2_avg = cv2.filter2D(imga, -1, averaging)
    cv2_sobel = cv2.filter2D(img2, -1, sobel_x)
    cv2_laplace = cv2.filter2D(img, -1, laplace)

    # Apply custom filter2D
    custom_avg = filter2D_custom(imga, averaging)
    custom_sobel = filter2D_custom(img2, sobel_x)
    custom_laplace = filter2D_custom(img, laplace)

    # Display comparisons
    titles = [
        'Original (avg)', 'cv2 Averaging', 'Custom Averaging',
        'Original (sobel)', 'cv2 Sobel X', 'Custom Sobel X',
        'Original (laplace)', 'cv2 Laplacian', 'Custom Laplacian'
    ]
    
    images = [
        imga, cv2_avg, custom_avg,
        img2, cv2_sobel, custom_sobel,
        img, cv2_laplace, custom_laplace
    ]

    plt.figure(figsize=(20, 12))
    for i in range(len(images)):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    image_path = 'image1.jpeg'  # Placeholder path
    main(image_path)
