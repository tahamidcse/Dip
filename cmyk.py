
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def generate_hsv(width=360, height=256, value=255):
 
    hsv_img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            hsv_img[y, x] = [x / width * 180, y, value]

    # Convert HSV to RGB
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    return rgb_img

def displayall():
    """
    Displays a 2D HSV color cube (Hue-Saturation slice with fixed Value).
    """
    color_cube = generate_hsv(width=360, height=256, value=255)

    plt.figure(figsize=(10, 4))
    plt.imshow(color_cube, aspect='auto')
    plt.title("HSV Color Cube Slice (Hue vs Saturation, Value = 255)")
    plt.xlabel("Hue (0° to 360°)")
    plt.ylabel("Saturation (0 to 255)")
    plt.xticks(np.linspace(0, 360, 7), labels=['0°','60°','120°','180°','240°','300°','360°'])
    plt.yticks(np.linspace(0, 255, 6))
    plt.grid(False)
    plt.show()

def generate(height=50, width=256):
    levels = np.linspace(0, 255, width, dtype=np.uint8)

    
    gray = np.dstack([np.tile(levels, (height, 1))]*3)

    
    red = np.dstack([np.tile(levels, (height, 1)),
                     np.zeros((height, width), dtype=np.uint8),
                     np.zeros((height, width), dtype=np.uint8)])

    
    green = np.dstack([np.zeros((height, width), dtype=np.uint8),
                       np.tile(levels, (height, 1)),
                       np.zeros((height, width), dtype=np.uint8)])

    
    blue = np.dstack([np.zeros((height, width), dtype=np.uint8),
                      np.zeros((height, width), dtype=np.uint8),
                      np.tile(levels, (height, 1))])

    cyan = np.dstack([np.zeros((height, width), dtype=np.uint8),
                      np.tile(levels, (height, 1)),
                      np.tile(levels, (height, 1))])

    
    magenta = np.dstack([np.tile(levels, (height, 1)),
                         np.zeros((height, width), dtype=np.uint8),
                         np.tile(levels, (height, 1))])


    yellow = np.dstack([np.tile(levels, (height, 1)),
                        np.tile(levels, (height, 1)),
                        np.zeros((height, width), dtype=np.uint8)])

    
    combined = np.vstack([gray, red, green, blue, cyan, magenta, yellow])

    return combined

def display():
    image = generate()
    plt.imshow(image)
    plt.title("Colors: Gray, Red, Green, Blue, Cyan, Magenta, Yellow")
    plt.axis('off')
    plt.show()

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
    displayall()
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
