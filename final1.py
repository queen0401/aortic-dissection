import cv2
import numpy as np
import pydicom as dicom
from matplotlib import pyplot as plt


# 从所有连通域中筛选出符合要求的连通域，可以根据要求读进不同的sample
def Filter_connectedComponents(num_labels, labels, stats, centers, image, sample, low_area, high_area):
    roi_list = []
    roi_area = []
    # print("in",num_labels)
    # 筛选连通域
    for t in range(1, num_labels, 1):  # 从1到num_labels，步长为1（num_labels：所有连通域的数目）
        x, y, w, h, area = stats[t]
        # stats：每一个标记的统计信息，是一个5列的矩阵
        # 每一行对应每个连通区域的外接矩形的x、y、width、height和面积，示例如下： 0 0 720 720 291805
        # 此步把label为t，即第t个连通域的矩阵信息分别赋给等号左侧的5个值
        # 由于连通域操作的只能是二值图像，所以返回的长方形区域就唯一确定了它中间的感兴趣的图像
        # 将每一个ROI与sample进行matchShape比对筛选
        roi = image[y:y + h, x:x + w]
        ret = cv2.matchShapes(sample, roi, 1, 0.0)
        if area < low_area or area > high_area or ret > 0.01:
            continue
        cx, cy = centers[t]  # 上一步没有被continue说明得到了一个符合筛选条件的ROI

        # 标出中心位置
        cv2.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
        # 画出外接矩形
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2, 8, 0)

        # 保存roi的坐标和长宽
        roi_list.append((x, y, w, h))  # roi_list是一个存储符合要求的roi的位置和长宽信息的矩阵
        roi_area.append(area)  # 所有符合要求的roi的面积都会存在roi_area列表中
    return roi_list, roi_area


# 把读入的sample处理成可以用于matchShape的格式
def tranform_Sample(sample):
    # 中值滤波
    sample = cv2.medianBlur(sample, 5)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    return sample

# get the front image
def getRoi(image):
    # 提取图中的所有8连通域
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=cv2.CV_32S)
    image = np.copy(image)
    # 准备用于形状比对的sample
    sample = tranform_Sample(cv2.imread("D:/python/pycharm/sample.jpg"))
    low_area = 350
    high_area = 900  # 设定面积大小的限制，需要的roi区域面积大小在350-900之间
    roi_list, roi_area = Filter_connectedComponents(num_labels, labels, stats, centers, image, sample, low_area, high_area)

    print("len = ", len(roi_list))
    if len(roi_list) == 0:  # 可能真假腔连起来或者只有真腔
        print("没能获得真腔和假腔，尝试寻找仅有的真腔。")
        sample_1 = tranform_Sample(cv2.imread("D:/python/pycharm/sample4.jpg"))
        low_area, high_area = 1000, 1500
        roi_list, roi_area = Filter_connectedComponents(num_labels, labels, stats, centers, image, sample_1,low_area, high_area)
        if len(roi_list) == 1:
            print("仅提取真腔成功，提取结果如下：")
            print(roi_area[0])
        else:
            print("roi提取遇到异常2")
    elif len(roi_list) == 2:  # 正常结果
        print("成功获得真假腔，提取结果如下：")
        print(roi_area[0], roi_area[1])
    else:
        print("roi提取遇到异常1")
    return num_labels, labels, image, roi_list, roi_area


def saveRoi(src, roi_list, roi_area):
    # src: 原图的copy
    # roi_list:保存的roi位置信息的矩阵列表
    # 对roi_list中的所有对象进行绘图
    for i in range(len(roi_list)):
        x, y, w, h = roi_list[i]
        roi = src[y:y+h, x:x+w]
        # cv2.imwrite("img%d.jpg" % i, roi)
        # cv2.imshow("img%d.jpg" % i, roi,)
        plt.subplot(1, 5, 2+i), plt.imshow(roi, 'gray'), plt.title('%d' % i)
        plt.xticks([]), plt.yticks([])
        print("No.%02d Finished! " % i)



if __name__ == '__main__':

    # 预处理
    ds = dicom.dcmread("D:/python/pycharm/fangjun1_2.dcm")
    img = ds.pixel_array * 50
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = np.array(img, dtype="uint8")

    im_in = img
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, im_th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
    # 经过高斯平滑、阈值处理、以及开运算得到opening的结果

    cv2.imshow("operated image",opening)
    plt.subplot(131), plt.imshow(opening, 'gray'), plt.title('operated image')
    plt.xticks([]), plt.yticks([])

    # 利用连通器寻找到需要提取的 roi
    num_labels, labels, image, roi_list, roi_area = getRoi(opening)


    # 保存roi
    saveRoi(opening, roi_list, roi_area)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
