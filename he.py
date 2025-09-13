import cv2
import matplotlib.pyplot as plt
import numpy as np
def round(x):
    y=int(x)
    if x<=(y+0.5):
       return y
    else:
         return y+1
def prepare_histogram(img, color_channel):
	#--- Prepare an array to hold the number of pixels
	L=256
	pixel_count = np.zeros((L,), dtype = np.uint)

	#--- Count number of pixels
	h, w = img.shape
	
	
	
	for i in range(h):
		for j in range(w):
			pixel_value = img[i,j]
			pixel_count[pixel_value] += 1
	
	
	



	#--- Plot histogram in two ways
	pdf=(pixel_count/(h*w))
	cdf=pdf.cumsum()
	x = np.arange(L)
	plt.figure(figsize = (20,20))
	plt.subplot(1, 2, 1)
	plt.plot(x, pixel_count, 'ro')
	plt.title('Non Equalized Histogram of ' + color_channel + ' Channel')
	
	plt.xlabel('Pixel Values')
	plt.ylabel('Number of Pixels')

	plt.subplot(1, 2, 2)
	plt.plot(x, cdf,'ro')
	
	plt.title('Implemented Equalized Histogram of ' + color_channel + ' Channel')
	plt.xlabel('Pixel Values')
	plt.ylabel('Number of Pixels')
	plt.show()
	plt.close()

	
def equalize_histogram(img, color_channel):
	#--- Prepare an array to hold the number of pixels
	L=256
	pixel_count = np.zeros((L,), dtype = np.uint)

	#--- Count number of pixels
	h, w = img.shape
	
	
	
	for i in range(h):
		for j in range(w):
			pixel_value = img[i,j]
			pixel_count[pixel_value] += 1
	
	pixels=h*w     
	pdf=(pixel_count/pixels)     
	csum=[]     
	psum=0     
	for i in pdf:         
	    psum+=i;         
	    csum.append(round(psum*(L-1)))

	cslen=len(csum)
	pc=np.zeros((L,), dtype = np.uint)
	for i in range(cslen):
		pc[csum[i]]+=pixel_count[i]
	   
	pdfe=(pc/pixels) 
	cdfe=pdfe.cumsum() 
	



	#--- Plot histogram in two ways
	x = np.arange(L)
	plt.figure(figsize = (20,20))
	plt.subplot(1, 2, 1)
	plt.plot(x, pc, 'ro')
	plt.title('Non Equalized Histogram of ' + color_channel + ' Channel')
	
	plt.xlabel('Pixel Values')
	plt.ylabel('Number of Pixels')

	plt.subplot(1, 2, 2)
	plt.plot(x, cdfe,'ro')
	
	plt.title('Implemented Equalized Histogram of ' + color_channel + ' Channel')
	plt.xlabel('Pixel Values')
	plt.ylabel('Number of Pixels')
	plt.show()
	plt.close()
def main():
    # Create a small grayscale image
    img = np.array([
        [4, 4, 4, 4, 4],
        [3, 4, 5, 4, 3],
        [3, 5, 6, 5, 3],
        [3, 4, 5, 4, 3],
        [4, 4, 4, 4, 4]], dtype=np.uint8)  # Ensure dtype is uint8
        
        

    # Apply histogram equalization
    img1=cv2.imread('Fig0316(1).png',0)
    img2=cv2.imread('Fig0316(2).tif',0)
    img3=cv2.imread('Fig0316(3).tif',0)

    img4=cv2.imread('Fig0316(4).tif',0)
        
        

    # Apply histogram equalization
    builtin_eq1 = cv2.equalizeHist(img1)
    builtin_eq2 = cv2.equalizeHist(img2)
    builtin_eq3 = cv2.equalizeHist(img3)
    builtin_eq4 = cv2.equalizeHist(img4)
    
    
   
   
    

    # Plot histogram of builtin equalized image
    prepare_histogram(builtin_eq1,'gray')
    
    
    # Plot histogram of implemented equalized image
    equalize_histogram(img1,'gray')



def display_imgset(img_set, color_set, title_set='', row=1, col=1):
    plt.figure(figsize=(8, 4))
    k = 1
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if k > len(img_set):
                break
            plt.subplot(row, col, k)
            img = img_set[k - 1]
            if len(img.shape) == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap=color_set[k - 1])
            if title_set[k - 1] != '':
                plt.title(title_set[k - 1])
            plt.axis('off')
            k += 1
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
   
