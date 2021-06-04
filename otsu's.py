import cv2
import numpy as np
from matplotlib import pyplot as plt

import pydicom as dicom

ds = dicom.dcmread("/dtest.dcm")
img = ds.pixel_array * 50


cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
img = np.array(img, dtype = "uint8")


# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
#（5,5）为高斯核的大小，0 为标准差
blur = cv2.GaussianBlur(img,(5,5),0)
# 阈值一定要设为0！
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print(ret2)

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

opening = cv2.GaussianBlur(opening,(5,5),0)

ret4,th4 = cv2.threshold(opening,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th2,
blur, 0, th3,
opening, 0, th4]

titles = ['Original Noisy Image','Histogram',"Otsu's Thresholding",
'Gaussian filtered Image','Histogram',"Otsu's Thresholding",
'Opening + Gaussian filtered image', 'Histogram', "Otsu's Thresholding"]


# 这里使用了pyplot 中画直方图的方法，plt.hist, 要注意的是它的参数是一维数组
# 所以这里使用了（numpy）ravel 方法，将多维数组转换成一维，也可以使用flatten 方法
#ndarray.flat 1-D iterator over an array.
#ndarray.flatten 1-D array copy of the elements of an array in row-major order.
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
