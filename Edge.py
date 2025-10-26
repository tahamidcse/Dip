import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Loading the Original Image
original_image = cv2.imread('./images/finger_print.tif', 0)
if original_image is None:
    print("Error: Could not load image")
    exit()

print(f"shape of the original image: {original_image.shape}")
plt.imshow(original_image, cmap='gray')
plt.title('original image')
plt.show()

img_arr = np.array(original_image)
print(f"min pixel intensity: {np.min(img_arr)}")
print(f"max pixel intensity: {np.max(img_arr)}")

def basic_global_thresholding(image, initial_threshold=128, epsilon=0.5):
    T = initial_threshold
    while True:
        # separate pixels
        higher_group = image[image > T]
        lower_group  = image[image <= T]
        
        # avoid division by zero
        if len(higher_group)==0 or len(lower_group)==0:
            break
        
        # calculate means of each group 
        higher_group_mean = np.mean(higher_group)
        lower_group_mean = np.mean(lower_group)
        
        # update threshold
        T_new = (higher_group_mean + lower_group_mean)/2
        
        # check for convergence
        if abs(T_new - T) < epsilon:
            break
        T = T_new
        
    # Apply the final threshold
    binary_image = np.where(image > T, 255, 0).astype(np.uint8)
    return binary_image, T    

# Apply basic global thresholding initially
binary_image, final_threshold = basic_global_thresholding(original_image)

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(binary_image, cmap='gray')
plt.title(f'Binary Image after Global Thresholding (Final T = {final_threshold:.2f})')

plt.tight_layout()
plt.show()

def threshold_segmentation_with_trackbar():
    # Create a window for threshold segmentation
    winname = 'Threshold Segmentation'
    cv2.namedWindow(winname)
    
    # Create trackbars for threshold parameters
    cv2.createTrackbar('Threshold', winname, 128, 255, lambda x: None)
    cv2.createTrackbar('Max Value', winname, 255, 255, lambda x: None)
    cv2.createTrackbar('Threshold Type', winname, 0, 4, lambda x: None)
    
    # Threshold type mapping
    threshold_types = {
        0: cv2.THRESH_BINARY,
        1: cv2.THRESH_BINARY_INV,
        2: cv2.THRESH_TRUNC,
        3: cv2.THRESH_TOZERO,
        4: cv2.THRESH_TOZERO_INV
    }
    
    threshold_type_names = {
        0: 'BINARY',
        1: 'BINARY_INV',
        2: 'TRUNC',
        3: 'TOZERO',
        4: 'TOZERO_INV'
    }
    
    print("Controls:")
    print("- Adjust 'Threshold' trackbar to change threshold value")
    print("- Adjust 'Max Value' trackbar to change maximum value")
    print("- Adjust 'Threshold Type' trackbar to change thresholding method")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset to global thresholding result")
    
    while True:
        # Get current trackbar positions
        threshold_value = cv2.getTrackbarPos('Threshold', winname)
        max_value = cv2.getTrackbarPos('Max Value', winname)
        threshold_type_idx = cv2.getTrackbarPos('Threshold Type', winname)
        
        # Apply thresholding
        _, binary_img = cv2.threshold(original_image, threshold_value, max_value, 
                                    threshold_types[threshold_type_idx])
        
        # Display the result
        cv2.imshow(winname, binary_img)
        
        # Display threshold info on image
        info_img = binary_img.copy()
        if len(info_img.shape) == 2:
            info_img = cv2.cvtColor(info_img, cv2.COLOR_GRAY2BGR)
        
        cv2.putText(info_img, f'Thresh: {threshold_value}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_img, f'Max: {max_value}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_img, f'Type: {threshold_type_names[threshold_type_idx]}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(winname, info_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset to global thresholding result
            cv2.setTrackbarPos('Threshold', winname, int(final_threshold))
            cv2.setTrackbarPos('Max Value', winname, 255)
            cv2.setTrackbarPos('Threshold Type', winname, 0)
    
    cv2.destroyAllWindows()

def cannyEdge():
    root = os.getcwd()
    imgPath = os.path.join(root, 'demolmages//tesla.jpg')
    img = cv2.imread(imgPath)
    
    if img is None:
        print("Error: Could not load image")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape
    scale = 1/5
    heightScale = int(height * scale)
    widthScale = int(width * scale)
    img = cv2.resize(img, (widthScale, heightScale), interpolation=cv2.INTER_LINEAR)

    winname = 'canny'
    cv2.namedWindow(winname)
    cv2.createTrackbar('minThres', winname, 0, 255, lambda x: None)
    cv2.createTrackbar('maxThres', winname, 0, 255, lambda x: None)

    # Set default values for trackbars
    cv2.setTrackbarPos('minThres', winname, 50)
    cv2.setTrackbarPos('maxThres', winname, 150)

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        minThres = cv2.getTrackbarPos('minThres', winname)
        maxThres = cv2.getTrackbarPos('maxThres', winname)
        cannyEdge = cv2.Canny(img, minThres, maxThres)
        cv2.imshow(winname, cannyEdge)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Run threshold segmentation with trackbar
    threshold_segmentation_with_trackbar()
    
    # Optionally run Canny edge detection
    # cannyEdge()
