#----------------------------------------------------
#--- To perform non-linear (e.g., Gamma, Logarithmic) 
#--- transformation on the pixel values of a digital 
#--- image.

#----------------------------------------------------
# 28.07.2025
#----------------------------------------------------

#--- Import necessary modules
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

def main():
    #--- Load a grayscale image
    img_path = '/home/bibrity/CSE_Courses/CSE4161_DIP/Images/paddy_field2.jpeg'
    img = cv2.imread(img_path, 0)
	
    #--- Perform Non-Linear Mapping
    gamma_transformation(img)
    logarithmic_transformation(img)

def logarithmic_transformation(img):
	#--- Perform logarithmic transformation
	h, w = img.shape
	log2_img = np.zeros((h,w), dtype = np.uint)
	log10_img = np.zeros((h,w), dtype = np.uint)
	c = 1
	for i in range(h):
		for j in range(w):
			log2_img[i, j] = c * math.log2(1 + img[i, j])
			log10_img[i, j] = c * math.log(1 + img[i, j])

	#--- Display images
	img_set = [img, log2_img, log10_img]
	title_set = ['Input_Img', 'Logarithmic_2_Img', 'Logarithmic_10_Img']
	color_set = ['gray', 'gray', 'gray']
	display_imgset(img_set, color_set, title_set = title_set, row = 1, col = 3)		

def gamma_transformation(img):
    c = 1
    gamma_list = [0.1, 0.5, 1, 2, 3]
    img_set = [img]
    title_set = ['Original_Img']
    color_set = ['gray']
    for gamma in gamma_list:
    	gamma_img = c * img ** gamma
    	img_set.append(gamma_img)
    	title_set.append('Gamma = ' + str(gamma))
    	color_set.append('gray')

    #--- Display images
    display_imgset(img_set, color_set, title_set = title_set, row = 2, col = 3)	
	
def display_imgset(img_set, color_set, title_set = '', row = 1, col = 1):
	plt.figure(figsize = (20, 20))
	k = 1
	for i in range(1, row + 1):
		for j in range(1, col + 1):
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
