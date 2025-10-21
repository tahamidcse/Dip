import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    img_path = '/home/bibrity/CSE_Courses/CSE4161_DIP/Images/rose1.jpeg'
    img1 = load_image(img_path)
    img_path = '/home/bibrity/CSE_Courses/CSE4161_DIP/Images/RGB_Ball.png'
    img2 = load_image(img_path)

    #--- Apply zero padding
    zero_padded_img = zero_padding(img1, img2)

    #--- Display images
    img_set = [img1, img2, zero_padded_img]
    title_set = ['Img1', 'Img2', 'Zero_Padding']
    color_set = ['gray', 'gray', 'gray']
    display_imgset(img_set, color_set, title_set, row = 1, col = 3)

def zero_padding(img1, img2):
    #--- Estimate height and width of two images
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    #--- Prepare a dark image having the same height and width
    #--- of the 2nd image. Here, we assume that the height and width
    #--- of the 2nd image are higher than the height and width of
    #--- the first image.
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
            zero_padded_img[i, j] = img1[h, w]
            w += 1
        h += 1
    return zero_padded_img

def load_image(img_path):
    #--- Load an image
    bgr_img = cv2.imread(img_path)
	
    #--- Convert the loaded BGR image into a grayscale image.
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    return gray_img

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
