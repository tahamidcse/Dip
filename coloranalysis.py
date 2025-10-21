#----------------------------------------------------
#--- To analyse different colored images.
#----------------------------------------------------

#----------------------------------------------------
# 01.08.2025
#----------------------------------------------------

#--- Import necessary modules
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    #--- Prepare different colored digital image
    h = 256
    w = 256

    black_img = np.zeros((h, w, 3), dtype = np.uint8)
    
    white_img = np.ones((h, w, 3), dtype = np.uint8) * 255

    red_img = np.zeros((h, w, 3), dtype = np.uint8)
    red_img[:, :, 0] = np.ones((h, w), dtype = np.uint8) * 255

    green_img = np.zeros((h, w, 3), dtype = np.uint8)
    green_img[:, :, 1] = np.ones((h, w), dtype = np.uint8) * 255

    blue_img = np.zeros((h, w, 3), dtype = np.uint8)
    blue_img[:, :, 2] = np.ones((h, w), dtype = np.uint8) * 255

    rgb_img = np.zeros((h, w, 3), dtype = np.uint8)
    for c in range(3):
    	for i in range(h):
    		for j in range(w):
    			rgb_img[i, j, c] = np.random.randint(0, 255) 

    gray_img = 0.299 * rgb_img[:, :, 0] + 0.587 * rgb_img[:, :, 1] + 0.114 * rgb_img[:, :, 2]
    gray_img = gray_img.astype(np.uint8)

    #--- Display images
    img_set = [black_img, white_img, red_img, green_img, blue_img, rgb_img, gray_img]
    color_set = ['', '', '', '', '', '', 'gray']
    title_set = ['Black','White', 'Red', 'Green', 'Blue', 'RGB', 'Gray']
    display_imgset(img_set, color_set, title_set = title_set, row = 2, col = 4)	
	
def display_imgset(img_set, color_set, title_set = '', row = 1, col = 1):
	plt.figure(figsize = (20, 20))
	k = 1
	n = len(img_set)
	for i in range(1, row + 1):
		for j in range(1, col + 1):
			if(i*j > n):
				break

			plt.subplot(row, col, k)
			img = img_set[k-1]
			if(len(img.shape) == 3):
				plt.imshow(img)
			else:
				plt.imshow(img, cmap = color_set[k-1])
			if(title_set[k-1] != ''):
				plt.title(title_set[k-1])

			k += 1
		
	plt.show()
	plt.close()

if __name__ == '__main__':
	main()
