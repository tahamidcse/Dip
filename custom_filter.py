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


def same_convolve(input_img, kernel):
    h1, w1 = input_img.shape
    h2 = kernel.shape[0] // 2
    w2 = kernel.shape[1] // 2

    #h2=h1+2
    #w2=w1+2

    zero_padded_img = np.zeros((h2, w2))
    

    #--- Estimate zero-padded border's height and width.
    d_h = int(abs(h2 - h1)/2)
    d_w = int(abs(w2 - w1)/2)

    #--- Put the fisrt image inside the dark image skipping
    #--- the border area.
    h = 0
    for i in range(d_h, h1+d_h):
        w = 0
        for j in range(d_w, w1 + d_w):
            zero_padded_img[i, j] = input_img[h, w]
            w += 1
        h += 1
    output_img=filter2D_custom(zero_padded_img,kernel)
    return output_img

def main(image_path):
    # Image paths
    imgpath="Fig0338(a)(blurry_moon).tif"
    imgpath2="Fig1022(a)(building_original).tif"
    imgp="Fig0333(a)(test_pattern_blurring_orig).tif"
    imgsx="Fig1022(a)(building_original).tif"
    imgsy="Fig1022(a)(building_original).tif"
    imgpx="Fig1022(a)(building_original).tif"
    imgpy="Fig1022(a)(building_original).tif"
    imgscx="Fig1022(a)(building_original).tif"
    imgscy="Fig1022(a)(building_original).tif"
    
    # Load grayscale images
    imga = cv2.imread(imgp, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imgpath2, cv2.IMREAD_GRAYSCALE)
        
    img_sy = cv2.imread(imgsy, cv2.IMREAD_GRAYSCALE)
    img_px = cv2.imread(imgpx, cv2.IMREAD_GRAYSCALE)
    img_py = cv2.imread(imgpy, cv2.IMREAD_GRAYSCALE)
    img_scx = cv2.imread(imgscx, cv2.IMREAD_GRAYSCALE)
    img_scy = cv2.imread(imgscy, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not read the image from {image_path}")
        return

    # Kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32)
    
    laplace = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]], dtype=np.float32)
    prewitty = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]], dtype=np.float32)
    prewittx = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]], dtype=np.float32)
    scarry = np.array([
        [-3,-10, -3],
        [0, 0, 0],
        [3, 10, 3]], dtype=np.float32)
    scarrx = np.array([
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]], dtype=np.float32)
    my_own=np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]], dtype=np.float32)

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
    cv2_sobel_y = cv2.filter2D(img_sy, -1, sobel_y)
    cv2_prewitt_x = cv2.filter2D(img_px, -1, prewittx)
    cv2_prewitt_y = cv2.filter2D(img_py, -1, prewitty)
    cv2_scharr_x = cv2.filter2D(img_scx, -1, scarrx)
    cv2_scharr_y = cv2.filter2D(img_scy, -1, scarry)

    # Apply filters using custom function
    custom_sobel_y = filter2D_custom(img_sy, sobel_y)
    custom_prewitt_x = filter2D_custom(img_px, prewittx)
    custom_prewitt_y = filter2D_custom(img_py, prewitty)
    custom_scharr_x = filter2D_custom(img_scx, scarrx)
    custom_scharr_y = filter2D_custom(img_scy, scarry)

    # Update titles and images
    titles += [
        'Original (sobel y)', 'cv2 Sobel Y', 'Custom Sobel Y',
        'Original (prewitt x)', 'cv2 Prewitt X', 'Custom Prewitt X',
        'Original (prewitt y)', 'cv2 Prewitt Y', 'Custom Prewitt Y',
        'Original (scharr x)', 'cv2 Scharr X', 'Custom Scharr X',
        'Original (scharr y)', 'cv2 Scharr Y', 'Custom Scharr Y'
    ]

    images += [
        img_sy, cv2_sobel_y, custom_sobel_y,
        img_px, cv2_prewitt_x, custom_prewitt_x,
        img_py, cv2_prewitt_y, custom_prewitt_y,
        img_scx, cv2_scharr_x, custom_scharr_x,
        img_scy, cv2_scharr_y, custom_scharr_y
    ]
        # Apply same_convolve filters
    same_avg = same_convolve(imga, averaging)
    same_sobel_x = same_convolve(img2, sobel_x)
    same_laplace = same_convolve(img, laplace)
    same_sobel_y = same_convolve(img_sy, sobel_y)
    same_prewitt_x = same_convolve(img_px, prewittx)
    same_prewitt_y = same_convolve(img_py, prewitty)
    same_scharr_x = same_convolve(img_scx, scarrx)
    same_scharr_y = same_convolve(img_scy, scarry)

    titles += [
        'Same Averaging',
        'Same Sobel X',
        'Same Laplacian',
        'Same Sobel Y',
        'Same Prewitt X',
        'Same Prewitt Y',
        'Same Scharr X',
        'Same Scharr Y'
    ]

    images += [
        same_avg,
        same_sobel_x,
        same_laplace,
        same_sobel_y,
        same_prewitt_x,
        same_prewitt_y,
        same_scharr_x,
        same_scharr_y
    ]



    # Adjust this based on number of images
    num_images = len(images)
    cols = 3
    rows = (num_images + cols - 1) // cols  # Ceiling division

    plt.figure(figsize=(cols * 6, rows * 4))  # Increase spacing

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=10)
        plt.axis('off')

plt.tight_layout()
plt.show()
plt.close()


if __name__ == "__main__":
    image_path = 'image1.jpeg'  # Placeholder path
    main(image_path)
