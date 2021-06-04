import cv2
import numpy as np
from matplotlib import pyplot as plt
import pydicom as dicom


ds = dicom.dcmread("/dtest.dcm")
img = ds.pixel_array

cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
img = np.array(img, dtype = "uint8")

blur = cv2.GaussianBlur(img,(5,5),0)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
