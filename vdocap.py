import cv2
import numpy as np
import matplotlib.pyplot as plt

def prepare_histogram(img, title):
    
    pixel_count = np.zeros(256, dtype=np.uint)
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            pixel_count[pixel_value] += 1
    
    plt.bar(np.arange(256), pixel_count)
    plt.title(title)
    plt.xlabel('Pixel Values')
    plt.ylabel('Number of Pixels')

def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        _, frame = cam.read()
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        inverse_img = 255 - gray_img
        shifted_img = np.clip(gray_img + 50, 0, 255).astype(np.uint8)

        
        c_log = 255 / np.log(1 + np.max(gray_img))
        log_img = c_log * np.log(1 + gray_img)
        log_img = np.array(log_img, dtype=np.uint8)

        
        gamma_value = 0.3
        gamma_img = np.array(255 * (gray_img / 255) ** gamma_value, dtype=np.uint8)
        
        
        plt.figure(figsize=(15, 15))

        
        plt.subplot(3, 3, 1)
        plt.imshow(rgb_img)
        plt.title('Original RGB')
        plt.axis('off')

        
        plt.subplot(3, 3, 2)
        prepare_histogram(gray_img, 'Original Histogram')

        
        plt.subplot(3, 3, 4)
        plt.imshow(inverse_img, cmap='gray')
        plt.title('Inverted Grayscale')
        plt.axis('off')

        
        plt.subplot(3, 3, 5)
        prepare_histogram(inverse_img, 'Inverted Histogram')

        
        plt.subplot(3, 3, 6)
        plt.imshow(log_img, cmap='gray')
        plt.title('Logarithmic Transformation')
        plt.axis('off')

        
        plt.subplot(3, 3, 7)
        prepare_histogram(log_img, 'Logarithmic Histogram')

        
        plt.subplot(3, 3, 8)
        plt.imshow(gamma_img, cmap='gray')
        plt.title(f'Gamma Correction (γ={gamma_value})')
        plt.axis('off')
        
        
        plt.subplot(3, 3, 9)
        prepare_histogram(gamma_img, 'Gamma Histogram')

        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)  

        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()
    plt.close('all')

if __name__ == "__main__":
    main()
