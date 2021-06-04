import cv2
import numpy as np
from matplotlib import pyplot as plt
import pydicom as dicom


ds=dicom.dcmread("/dtest.dcm")
img = ds.pixel_array

ret, thresh = cv2.threshold(img,187,255,cv2.THRESH_BINARY)


plt.subplot(2,3,1),plt.imshow(thresh,'gray')
plt.title("result")
plt.xticks([]),plt.yticks([])
plt.show()
