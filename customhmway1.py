import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.exposure import match_histograms
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
	print(pixel_count)
	pixels=h*w
	pdf=pixel_count/pixels
	cdf=pdf.cumsum()
	



	#--- Plot histogram in two ways
	
	pdf*=pixels
	return cdf,pdf
def plot_histogram(pixel_count,color_channel):
	#--- Prepare an array to hold the number of pixels
	L=256
	
	



	#--- Plot histogram in two ways
	x = np.arange(L)
	plt.figure(figsize = (20,20))
	plt.subplot(1, 2, 1)
	plt.plot(x, pixel_count, 'ro')
	plt.title('Built in Histogram of ' + color_channel + ' Channel')
	plt.xlabel('Pixel Values')
	plt.ylabel('Number of Pixels')
	plt.show()
	plt.close()
	

	
	

def main():
   
        
        

    # Apply histogram equalization
    img1=cv2.imread('Fig0309(a)(washed_out_aerial_image).tif',0)
   
    spec=cv2.imread('Fig0316(3).tif',0)

 
    cdfg,pdfg=prepare_histogram(img1,'gray')
    cdfs,_=prepare_histogram(spec,'gray')
    mini=float('inf')
    midx=0
    glevel=[]
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mini=float('inf')
        midx=0
        
        for j in range(256):
            diff=np.abs(cdfg[i]-cdfs[j])
            if diff<mini:
               mini=diff
               midx=j
        mapping[i] = midx
    
    #matched_custom = mapping[img1]
    h,w=img1.shape
    mimg=np.zeros_like(img1)
    for i in range(h):
        for j in range(w):
            mimg[i][j]=mapping[img1[i][j]]

    # --- Built-in Histogram Matching using skimage ---
    matched_builtin = match_histograms(img1, spec, channel_axis=None)

    # --- Histograms ---
    _, pdf_original = prepare_histogram(img1,'original')
    _, pdf_custom = prepare_histogram(mimg.astype(np.uint8),'custom matched')
    _, pdf_builtin = prepare_histogram(matched_builtin.astype(np.uint8),'scikit matched')

    # --- Plot Histograms ---
    plot_histogram(pdf_original, 'Original Image Histogram')
    plot_histogram(pdf_custom,'Custom Matched Histogram')
    plot_histogram(pdf_builtin, 'Built-in Matched Histogram')
    
        
            
        
    
        
        



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
   
