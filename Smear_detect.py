import cv2
import imutils
import sys
import glob
from matplotlib import pyplot as plt
import numpy as np
import time

a = sys.argv
if len(a) < 2:
    print("USAGE: python Smear_detect.py <Images path>/*.jpg")
    exit(1)
try:
    now = time.strftime("%c")
    ## date and time representation
    print("Detection started at: " + time.strftime("%c"))
    print("")
    #Getting the directory of the images
    files = glob.glob(a[1])
    img_avg = np.zeros((500,500),np.float)
    for im in files:
        input = cv2.imread(im)
        input = imutils.resize(input, width=500)
        input = cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)
        input = cv2.equalizeHist(input)
        img_avg = img_avg + input

    img_avg = img_avg/len(files)
    img_avg = np.array(np.round(img_avg), dtype=np.uint8)

    cv2.imwrite("Average.jpg", img_avg)
except FileNotFoundError:
    print("Enter proper Directory")
#Reading the averaged image
img = cv2.imread('Average.jpg',0)

img1 = cv2.imread(glob.glob(a[1])[41])
img2 = cv2.imread(glob.glob(a[1])[41])

img2 = imutils.resize(img2,width=500)
img1 = imutils.resize(img1,width=500)

img3 = cv2.imread('Average.jpg')

#Adding adaptive threshold
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,4)

#softening the image by removing salt-pepper noise
th3 = cv2.medianBlur(th3,29)
warped = th3.astype("uint8") * 255

#Detecting the edges in the image
edge_detected_image = cv2.Canny(warped, 9,50,apertureSize=5,L2gradient=True)

#Detecting the Contours
image, contours, hierarchy = cv2.findContours(edge_detected_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

plt.imshow(sobelx,cmap = 'gray')
plt.axis('off')
plt.savefig('Gradient.jpg', bbox_inches='tight', pad_inches = 0)

list = []
for i in contours:
    list.append(i)
mask = np.zeros((500,500,1),np.float)
if len(list) > 0:
    img = cv2.drawContours(img1, contours, -1, (0,255,0), 3)
    k = cv2.drawContours(mask, contours, -1, (255, 255, 255), 15)
    img3 = cv2.drawContours(img3, contours, -1, (0, 255, 0), 3)

    cv2.imwrite('SmearOnAverageImage.jpg',img3)
    cv2.imwrite('MaskedImage.jpg',mask)

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Resultant Image with smear'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(img2,cmap = 'gray')
    plt.title('Original Image without smear detected'), plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.savefig('FinalResult.jpg')
    plt.show()
    print("Smear Detected. Result in FinalResult.jpg")
else:
    print("Smear not detected")