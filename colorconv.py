#----------------------------------------------------
#--- To convert color of a digital image.
#----------------------------------------------------
#--- Sangeeta Biswas, Ph.D.
#--- Associate Professor
#--- Department of Computer Science and Engineering
#--- University of Rajshahi
#--- Rajshahi-6205, Bangladesh
#----------------------------------------------------
# 29.07.2025
#----------------------------------------------------

#--- Import necessary modules
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    #--- Load a digital image
    img_path = '/home/bibrity/CSE_Courses/CSE4161_DIP/Images/paddy_field2.jpeg'
    bgr_img = cv2.imread(img_path)

    #--- Color conversion by built-in-functions
    gray_img1 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    rgb_img1 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    #--- Split color channel
    blue_channel = bgr_img[:, :, 0]
    green_channel = bgr_img[:, :, 1]
    red_channel = bgr_img[:, :, 2]
	
    #--- Color conversion manually
    gray_img2 = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
    gray_img2 = gray_img2.astype(np.uint8)
    h, w, _ = bgr_img.shape
    rgb_img2 = np.zeros((h, w, 3), dtype = np.uint8)
    rgb_img2[:, :, 0] = red_channel
    rgb_img2[:, :, 1] = green_channel
    rgb_img2[:, :, 2] = blue_channel    

    #--- Display images
    img_set = [bgr_img, gray_img1, rgb_img1, gray_img2, rgb_img2]
    color_set = ['', 'gray', '', 'gray', '']
    title_set = ['BGR', 'CV2_Grayscale', 'CV2_RGB', 'Manual_Grayscale', 'Manual_RGB']
    display_imgset(img_set, color_set, title_set = title_set, row = 2, col = 3)	
	
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
