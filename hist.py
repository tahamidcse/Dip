#----------------------------------------------------
#--- To generate the histogram of a digital image.
#----------------------------------------------------
#--- Sangeeta Biswas, Ph.D.
#--- Associate Professor
#--- Department of Computer Science and Engineering
#--- University of Rajshahi
#--- Rajshahi-6205, Bangladesh
#----------------------------------------------------
# 22.7.2025
#----------------------------------------------------

#--- Import necessary modules
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    #--- Load an image
    img_path = '/home/bibrity/CSE_Courses/CSE4161_DIP/Images/rose1.jpeg'
    bgr_img = cv2.imread(img_path)
	
    #--- Change channel order to cope with Matplotlib requirement.
    #--- OpenCV loaded images in BGR (Blue, Green, Red Channel) order.
    #--- Matplotlib handle images in RGB order.
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    #--- Separate different channels
    red_img = rgb_img[:, :, 0]
    green_img = rgb_img[:, :, 1]
    blue_img = rgb_img[:, :, 2]

    #--- Display images
    img_set = [rgb_img, red_img, green_img, blue_img]
    title_set = ['RGB', 'Red', 'Green', 'Blue']
    color_set = ['', 'Reds', 'Greens', 'Blues']
    display_imgset(img_set, color_set, title_set, row = 2, col = 2)

    #--- Prepare histogram for each color channel separately
    prepare_histogram(red_img, 'Red')
    prepare_histogram(green_img, 'Green')
    prepare_histogram(blue_img, 'Blue')

def prepare_histogram(img, color_channel):
	#--- Prepare an array to hold the number of pixels
	pixel_count = np.zeros((256,), dtype = np.uint)

	#--- Count number of pixels
	h, w = img.shape
	for i in range(h):
		for j in range(w):
			pixel_value = img[i,j]
			pixel_count[pixel_value] += 1
	print(pixel_count)

	#--- Plot histogram in two ways
	x = np.arange(256)
	plt.figure(figsize = (20,20))
	plt.subplot(1, 2, 1)
	plt.plot(x, pixel_count, 'ro')
	plt.title('Histogram of ' + color_channel + ' Channel')
	plt.xlabel('Pixel Values')
	plt.ylabel('Number of Pixels')

	plt.subplot(1, 2, 2)
	plt.bar(x, pixel_count)
	plt.title('Histogram of ' + color_channel + ' Channel')
	plt.xlabel('Pixel Values')
	plt.ylabel('Number of Pixels')
	plt.show()
	plt.close()
	
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
