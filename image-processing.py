import cv2
import numpy as np
import matplotlib.pyplot as plt

#   ~ SUBSECTION 1 ~

#load selfie and convert it from bgr to hls
img = cv2.imread(r"C:\Users\you\Documents\photo.jpeg")
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

#split channels and apply clahe on the L channel
H, L, S = cv2.split(hls)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
luminance_clahe = clahe.apply(L)

#merge channels and convert back to BGR to create the equalized photo
hls_clahe = cv2.merge((H,luminance_clahe,S))
imgEqualized = cv2.cvtColor(hls_clahe, cv2.COLOR_HLS2BGR)

#display the original image, the changes and the histograms
plt.figure(figsize=(12,6))
plt.subplot(221), plt.imshow(img[...,::-1]), plt.title('Original image')
plt.subplot(222), plt.hist(L.ravel(), bins=256, range=[0,256], color='orange', alpha=0.6), plt.title('Original Histogram - L')
plt.subplot(223), plt.imshow(imgEqualized[...,::-1]), plt.title('Equalized image (clahe)')
plt.subplot(224), plt.hist(luminance_clahe.ravel(), bins=256, range=[0,256], color='pink', alpha=0.6), plt.title('Equalized Histogram - luminance_clahe')
plt.show()

######################################################################################################################################################################################################

#   ~ SUBSECTION 2 ~

#convert image to grayscale
imgGray = cv2.cvtColor(imgEqualized, cv2.COLOR_BGR2GRAY)
#apply Canny to detect edges
edge = cv2.Canny(imgGray, 125,150)
#obtain black edges on white backgorund
edge_inverted = cv2.bitwise_not(edge)

#make the image smooth
imgBlurred = cv2.bilateralFilter(imgEqualized, d=9, sigmaColor=250, sigmaSpace=250)
#combine the equalized and blurred images
imgCartoon = cv2.bitwise_and(imgEqualized, imgBlurred, mask=edge_inverted)

#display
plt.figure()
plt.subplot(231), plt.imshow(imgEqualized[...,::-1]), plt.title('Equalized image (clahe)')
plt.subplot(232), plt.imshow(imgGray, cmap='gray'), plt.title('Grayscale image')
plt.subplot(233), plt.imshow(edge, cmap='gray'), plt.title('Edge detection')
plt.subplot(234), plt.imshow(edge_inverted, cmap='gray'), plt.title('Inverted edges')
plt.subplot(235), plt.imshow(imgBlurred[...,::-1]), plt.title('Smoothed image')
plt.subplot(236), plt.imshow(imgCartoon[...,::-1]), plt.title('Cartoonified image')
plt.show()

####################################################################################################################################################################################################

#   ~ SUBSECTION 3 ~

#load background image
background = cv2.imread(r"C:\Users\you\Documents\landscape.jpg")

#resize the cartoonized image
imgResize = cv2.resize(imgCartoon,(200,200))
#create the mask - same size as the cartoonized resized image
mask = np.zeros(imgResize.shape[:2], dtype=np.uint8)

#define the coordinates of the face contour
coord = np.array([[47,100],[54,44],[134,36],[140,120],[69,137],[69,130]], dtype=np.int32)
#draw the mask and fill it with white colour
cv2.fillPoly(mask,[coord],255)

#pick a point where to put the cartoonized image on the background
center = (315,482)
#merge the mask with the background
imgClone = cv2.seamlessClone(imgResize, background, mask, center,cv2.NORMAL_CLONE)

#display
plt.figure()
plt.subplot(121), plt.imshow(imgResize[...,::-1]), plt.title('Cartoonified image')
plt.subplot(122), plt.imshow(mask, cmap='gray'), plt.title('Face mask')
plt.show()

plt.figure()
plt.subplot(121), plt.imshow(background[...,::-1]), plt.title('Backgorund')
plt.subplot(122), plt.imshow(imgClone[...,::-1]), plt.title('Face cloned on background')
plt.show()
