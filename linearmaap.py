#----------------------------------------------------
#--- To perform linear operation on the pixel values
#--- of a digital image.
#----------------------------------------------------

#----------------------------------------------------
# 24.07.2025
#----------------------------------------------------

#--- Import necessary modules
import cv2
import matplotlib.pyplot as plt

def main():
    #--- Load an image
    img_path = '/home/bibrity/CSE_Courses/CSE4161_DIP/Images/rose1.jpeg'
    bgr_img = cv2.imread(img_path)
	
    #--- Change channel order to cope with Matplotlib requirement.
    #--- OpenCV loaded images in BGR (Blue, Green, Red Channel) order.
    #--- Matplotlib handle images in RGB order.
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    #--- Image Inversion/ Image negation
    inverse_img = 255 - rgb_img

    #--- Shift image pixel to the right side
    c = 10
    shifted_img = rgb_img + c

    #--- Display images
    img_set = [rgb_img, inverse_img, shifted_img]
    title_set = ['RGB_Img', 'Inverse_Img', 'Shifted_Img']
    color_set = []
    display_imgset(img_set, color_set, title_set, row = 1, col = 3)
	
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
