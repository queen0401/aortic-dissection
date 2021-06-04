import cv2
import numpy as np
from matplotlib import pyplot as plt
import pydicom as dicom


ds = dicom.dcmread("/dtest.dcm")
img = ds.pixel_array

# 中值滤波
img = cv2.medianBlur(img,5)

cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
img = np.array(img, dtype = "uint8")

edges = cv2.Canny(img,30,90)


plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
