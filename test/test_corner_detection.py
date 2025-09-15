#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from tkinter import filedialog
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def askopenimagefilename():
    return filedialog.askopenfilename(
        title='select an image file to open',
        initialdir='./calibration_images',
        filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp;*.tiff'), ('All files', '*.*')],
    )


def detect_corners():
    path_image = askopenimagefilename()
    print(f'Opening file {path_image!r} ...')

    if not path_image:
        exit()

    # 读取图像
    img_original = cv.imread(path_image)
    # 将图像转换为灰度图像
    img_grayscale = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    img_grayscale = np.float32(img_grayscale)
    # 查找角点
    dst = cv.cornerHarris(img_grayscale, 2, 3, .04)
    dst = cv.dilate(dst, None)

    corner_threshold = .01 * dst.max()
    nms_size = 5  # NMS 的邻域大小
    nms_kernel = cv.getStructuringElement(cv.MORPH_RECT, (nms_size, nms_size))  # 创建一个内核，用来寻找局部最大值
    nms_local_max = cv.dilate(dst, nms_kernel)  # 与原始 dst 进行比较，只保留局部最大值对应的点
    nms_mask = (dst == nms_local_max) & (dst > corner_threshold)
    where_threshold = np.where(nms_mask)
    corner_coordinates = np.column_stack(where_threshold)

    img_original[where_threshold] = [0, 0, 255]
    cv.imshow('dst', img_original)

    # 4. 打印或保存角点坐标
    print(f"检测到 {len(corner_coordinates)} 个角点")
    for i, corner in enumerate(corner_coordinates):
        print(f"角点 {i+1}: x={corner[0]:.2f}, y={corner[1]:.2f}")

if __name__ == '__main__':
    # 打印 python 版本、opencv 版本、numpy 版本、matplotlib 版本
    print(f'Python version: {sys.version}')
    print(f'\tGIL enabled: {sys._is_gil_enabled()}')
    print(f'OpenCV version: {cv.__version__}')
    print(f'Numpy version: {np.__version__}')
    print(f'Matplotlib version: {matplotlib.__version__}')

    detect_corners()

    while True:
        key_event = 0xff & cv.waitKey(1)
        if key_event == ord('q'):
            cv.destroyAllWindows()
            exit()
        elif key_event == ord('o'):
            detect_corners()
