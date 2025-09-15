import numpy as np
import cv2
import matplotlib.pyplot as plt
#built in + own salt and pepper noise remove
def main():
    img=cv2.imread('FigP0433(left)(DIP_image).tif',0)
    ker=np.ones((3,3),dtype=np.uint8)
    ker2=np.ones((5,5),dtype=np.uint8)
    eimg=cv2.erode(img,ker,iterations=2)
    dimg=cv2.dilate(eimg,ker,iterations=2)
    cv2.imshow('Original Image', img)
    cv2.imshow('Eroded Image', mimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__=="__main__":
   main()
