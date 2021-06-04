import cv2
import numpy as np
import pydicom as dicom


ds = dicom.dcmread("D:/python/pycharm/dtest_tears.dcm")
img = ds.pixel_array * 50

# 中值滤波
img = cv2.medianBlur(img,5)

cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
img = np.array(img, dtype = "uint8")

cv2.imshow("original image",img)

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow("operated image",opening)

cv2.waitKey()
cv2.destroyAllWindows()
