import cv2 as cv
import os

def cannyEdge():
    root = os.getcwd()
    imgPath = os.path.join(root, 'demolmages//tesla.jpg')
    img = cv.imread(imgPath)
    
    if img is None:
        print("Error: Could not load image")
        return
        
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height, width, _ = img.shape
    scale = 1/5
    heightScale = int(height * scale)
    widthScale = int(width * scale)
    img = cv.resize(img, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)

    def callback(x):
        pass  # Empty callback function for trackbar

    winname = 'canny'
    cv.namedWindow(winname)
    cv.createTrackbar('minThres', winname, 0, 255, callback)
    cv.createTrackbar('maxThres', winname, 0, 255, callback)

    # Set default values for trackbars
    cv.setTrackbarPos('minThres', winname, 50)
    cv.setTrackbarPos('maxThres', winname, 150)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        minThres = cv.getTrackbarPos('minThres', winname)
        maxThres = cv.getTrackbarPos('maxThres', winname)
        cannyEdge = cv.Canny(img, minThres, maxThres)
        cv.imshow(winname, cannyEdge)

    cv.destroyAllWindows()

if __name__ == '__main__':
    cannyEdge()
