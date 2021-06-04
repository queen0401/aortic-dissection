import cv2
import numpy as np
from matplotlib import pyplot as plt
import pydicom as dicom

ds = dicom.dcmread("/dtest.dcm")
img = ds.pixel_array

# 中值滤波
img = cv2.medianBlur(img,5)

cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
img = np.array(img, dtype = "uint8")


ret,th1 = cv2.threshold(img,187,255,cv2.THRESH_BINARY)
#11 为Block size, 2 为C 值
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 187)',
'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
