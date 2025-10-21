#----------------------------------------------------
#--- To perform Bit-Plane Slicing, i.e., decomposing 
#--- a digital image into a series of binary images, 
#--- each representing a specific bit position of 
#--- the pixel values.
#----------------------------------------------------
#--- Sangeeta Biswas, Ph.D.
#--- Associate Professor
#--- Department of Computer Science and Engineering
#--- University of Rajshahi
#--- Rajshahi-6205, Bangladesh
#----------------------------------------------------
# 02.08.2025
#----------------------------------------------------

#--- Import necessary modules
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    #--- Load an image
    img_path = '/home/bibrity/CSE_Courses/CSE4161_DIP/Images/paddy_field1.jpeg'
    gray_img = cv2.imread(img_path, 0)

    #--- Slice an 8-bit grayscale image into 8 planes in two ways
    bit_planes = bit_plane_slicing_1stWay(gray_img)
    #bit_planes = bit_plane_slicing_2ndWay(gray_img)

    #--- Combined different bit-planes to reconstruct the grayscale image
    cmd_planes123 = bit_planes[0] + bit_planes[1]  + bit_planes[2]
    cmd_planes234 = bit_planes[1] + bit_planes[2]  + bit_planes[3]
    cmd_planes456 = bit_planes[3] + bit_planes[4]  + bit_planes[5]
    cmd_planes567 = bit_planes[4] + bit_planes[5]  + bit_planes[6] 
    cmd_planes678 = bit_planes[5] + bit_planes[6]  + bit_planes[7] 
    reconstructed_img = bit_planes[0] + bit_planes[1] + bit_planes[2] +\
    					bit_planes[3] + bit_planes[4] + bit_planes[5] +\
    					bit_planes[6] + bit_planes[7]

    #--- Check whether Reconsytructed image and Original image are the same.
    loss = np.sum(gray_img - reconstructed_img)
    print(loss)
    if (loss == 0):
    	print('Lossless reconstruction was done...')

    #--- Display images
    img_set = [gray_img]
    img_set += bit_planes
    img_set += [
    	cmd_planes123, cmd_planes234, cmd_planes456, 
    	cmd_planes567, cmd_planes678, reconstructed_img
    ]

    title_set = [
    	'Original Image', 'Plane_1', 'Plane_2', 'Plane_3', 'Plane_4',
    	'Plane_5', 'Plane_6', 'Plane_7','Plane_8', 'Plane_1+2+3',
    	'Plane_2+3+4', 'Plane_4+5+6', 'Plane_5+6+7', 'Plane_6+7+8',
    	'All_Planes'
    ]
    color_set = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 
    			 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']
   
    display_imgset(img_set, color_set, title_set, row = 3, col = 5)

def bit_plane_slicing_2ndWay(gray_img):
	bit_planes = []
	for i in range(8):
		#--- Step-1: Right-shift the pixel value by the bit plane number 
		#--- to move the target bit to the LSB position.
		bit_shifted_img = gray_img >> i

		#--- Step-2: Perform a bitwise AND operation with 1 to extract
		#--- the value of that LSB (either 0 or 1).
		LSB_bit = bit_shifted_img & 1

		#--- Step-3: Multiply the extracted  LSB bit by 2**i to create
		#--- a grayscale image
		sliced_img = LSB_bit * (2**i)

		bit_planes.append(sliced_img)

	return bit_planes

def bit_plane_slicing_1stWay(gray_img):
	binary_mask_set = [
		0b00000001, 0b00000010, 0b00000100, 0b00001000,
		0b00010000, 0b00100000, 0b01000000, 0b10000000
	]
	bit_planes = []
	for i in range(8):
		sliced_img = gray_img & binary_mask_set[i]
		bit_planes.append(sliced_img)

	return bit_planes

def display_imgset(img_set, color_set, title_set = '', row = 1, col = 1):
	plt.figure(figsize = (20, 20))
	k = 1
	n = len(img_set)
	for i in range(1, row + 1):
		for j in range(1, col + 1):
			if (k > n):
				break
			plt.subplot(row, col, k)
			plt.axis('off')
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
