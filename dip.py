
import cv2
import matplotlib.pyplot as plt
import numpy as np

def cyan_level():
  img_set=[]
  color_set=[]
  h=256
  w=256
  for i in range(6,9):
    red= np.ones((h, w), dtype=np.uint8) * 0
    green= np.ones((h, w), dtype = np.uint8)*((2**i)-1)
    blue= np.ones((h, w), dtype = np.uint8)*((2**i)-1)
    rgb=np.dstack((red,blue,green))
    img_set.append(rgb)
    color_set.append('');
  display_imgset2(img_set, color_set, row = 1, col = 3)


def magenta_level():
  img_set=[]
  color_set=[]
  h=256
  w=256
  for i in range(6,9):
    red= np.ones((h, w), dtype=np.uint8) *((2**i)-1)
    green= np.ones((h, w), dtype = np.uint8)*0
    blue= np.ones((h, w), dtype = np.uint8)*((2**i)-1)
    rgb=np.dstack((red,blue,green))
    img_set.append(rgb)
    color_set.append('');
  display_imgset2(img_set, color_set, row = 1, col = 3)
def yellow_level():
  img_set=[]
  color_set=[]
  h=256
  w=256
  for i in range(6,9):
    red= np.ones((h, w), dtype=np.uint8) * ((2**i)-1)
    green= np.ones((h, w), dtype = np.uint8)*((2**i)-1)
    blue= np.ones((h, w), dtype = np.uint8)*(0)
    rgb=np.dstack((red,blue,green))
    img_set.append(rgb)
    color_set.append('');
  display_imgset2(img_set, color_set, row = 1, col = 3)




def white_level():



  img_set=[]
  color_set=[]
  h=256
  w=256

  for i in range(6,9):
    wimg = np.ones((h, w, 3), dtype = np.uint8) *((2**i)-1)

    

    img_set.append(wimg)
    color_set.append('');
  display_imgset2(img_set, color_set, row = 1, col = 3)
def red_level():



  img_set=[]
  color_set=[]
  h=256
  w=256

  for i in range(6,9):
    rimg = np.zeros((h, w, 3), dtype = np.uint8)
    rimg[:, :, 0] = np.ones((h, w), dtype = np.uint8) *((2**i)-1)

    img_set.append(rimg)
    color_set.append('');
  display_imgset2(img_set, color_set, row = 1, col = 3)
def green_level():



  img_set=[]
  color_set=[]
  h=256
  w=256

  for i in range(6,9):
    rimg = np.zeros((h, w, 3), dtype = np.uint8)
    rimg[:, :, 1] = np.ones((h, w), dtype = np.uint8) *((2**i)-1)

    img_set.append(rimg)
    color_set.append('');
  display_imgset2(img_set, color_set, row = 1, col = 3)
def blue_level():



  img_set=[]
  color_set=[]
  h=256
  w=256

  for i in range(6,9):
    rimg = np.zeros((h, w, 3), dtype = np.uint8)
    rimg[:, :, 2] = np.ones((h, w), dtype = np.uint8) *((2**i)-1)

    img_set.append(rimg)
    color_set.append('');
  display_imgset2(img_set, color_set, row = 1, col = 3)


def main():

    #--- Prepare different colored digital image
    white_level()
    red_level()
    green_level()
    blue_level()
    cyan_level()
    magenta_level()
    yellow_level()




    





def display_imgset2(img_set, color_set, row = 1, col = 1):
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


			k += 1

	plt.show()
	plt.close()

if __name__ == '__main__':
	main()
