import cv2
import numpy as np
import matplotlib.pyplot as plt
def main(image_path):
    # Read the image in grayscale
    imgpath="Fig0338(a)(blurry_moon).tif"
    imgpath2="Fig1022(a)(building_original).tif"
    imgp="doll.jpg"#"Fig0334(a)(hubble-original).tif"
    imga = cv2.imread(imgp, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imgpath2, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not read the image from {image_path}")
        return

    # Get image shape
    r, c = img.shape

    # Add Gaussian noise


    # Define the Sobel kernel (Y-direction)
    laplace = np.array([
        [0, 1, -0],
        [ 1,  -4,  1],
        [ 0,  1,0  ]
    ], dtype=np.float32)

    sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]], dtype=np.float32)
    sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]], dtype=np.float32)
    averaging = (1/9)*np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]], dtype=np.float32)












    prewitt_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]], dtype=np.float32)

    prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]], dtype=np.float32)




    # Apply the kernel using cv2.filter2D
    imgsx=img2.copy()
    imgsy=img2.copy()
    imgpx=img2.copy()
    imgpy=img2.copy()
    imglap=img
    
    
    filtered_imga = cv2.filter2D(imga, -1, averaging)
    filtered_imgsx = cv2.filter2D(imgsx, -1, sobel_x)
    filtered_imgsy = cv2.filter2D(imgsy, -1, sobel_y)
    filtered_imgpx = cv2.filter2D(imgpx, -1, prewitt_x)
    filtered_imgpy = cv2.filter2D(imgpy, -1, prewitt_y)
    filtered_imglap = cv2.filter2D(imglap, -1, laplace)
    

    # Display the images
    titles = ['before average','before laplace','before edge detect', 'Averaging', 'Sobel X', 'Sobel Y', 'Prewitt X', 'Prewitt Y', 'Laplacian']
    images = [imga, imglap,imgsx,filtered_imga, filtered_imgsx, filtered_imgsy, filtered_imgpx, filtered_imgpy, filtered_imglap]
    plt.figure(figsize=(20,20))
    for i in range(len(images)):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    image_path = 'image1.jpeg'  # Replace with your image path
    main(image_path)

