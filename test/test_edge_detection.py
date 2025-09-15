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


def plot_histogram_of_area_of_contours(data):
    area_of_contours = [
        area
        for contour in data
        if (area := cv.contourArea(contour)) > 500
            and area < 1000
    ]
    count_of_contours = len(area_of_contours)
    max_area_of_contours = max(area_of_contours) if area_of_contours else 0
    # print('Max contour area:', max_area_of_contours, type(max_area_of_contours))
    bin_size = 10
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(
        area_of_contours,
        bins=int(max_area_of_contours) // bin_size + 1,
        color='skyblue',
        edgecolor='black',
        alpha=.7,
    )
    plt.xlabel('Contour Area')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Contour Areas (Total: {count_of_contours})')
    plt.show()


if __name__ == '__main__':
    # 打印 python 版本、opencv 版本、numpy 版本、matplotlib 版本
    print(f'Python version: {sys.version}')
    print(f'\tGIL enabled: {sys._is_gil_enabled()}')
    print(f'OpenCV version: {cv.__version__}')
    print(f'Numpy version: {np.__version__}')
    print(f'Matplotlib version: {matplotlib.__version__}')

    path_image = askopenimagefilename()
    print(f'Opening file {path_image!r} ...')

    if not path_image:
        exit()

    # 读取图像
    img_original = cv.imread(path_image)
    # 将图像转换为灰度图像
    img_grayscale = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    # 使用高斯模糊对图像进行预处理，让它变得更平滑，减少噪声干扰
    img_blurred = cv.GaussianBlur(img_grayscale, (5, 5), 1.5)
    # 归一化
    img_normalized = cv.normalize(img_blurred, None, 0, 255, cv.NORM_MINMAX)

    # 阈值分割
    _, img_thresholded = cv.threshold(img_grayscale, 80, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(img_thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # contour_counter = 0
    # for contour in contours:
    #     area = cv.contourArea(contour)
    #     if area < 100 or area > 10000:
    #         continue
    #     contour_counter += 1
    #     print(f'contour[{contour_counter}] Area=', area)

    # 计算已检测的轮廓的面积的直方图
    plot_histogram_of_area_of_contours(contours)

    # # 使用 Sobel 算子进行边缘检测
    # sobel_x = cv.Sobel(img_normalized, cv.CV_64F, 1, 0, ksize=3)
    # sobel_y = cv.Sobel(img_normalized, cv.CV_64F, 0, 1, ksize=3)
    # # 计算梯度幅值
    # sobel_edges = cv.addWeighted(
    #     cv.convertScaleAbs(sobel_x), .5,
    #     cv.convertScaleAbs(sobel_y), .5,
    #     0
    # )

    # 使用 Canny 边缘检测算法
    # canny_edges = cv.Canny(img_normalized, 50, 150)

    # cv.imshow('Original Image', img_original)
    # cv.imshow('Grayscale Image', img_grayscale)
    # cv.imshow('Sobel X', sobel_x)
    # cv.imshow('Sobel Y', sobel_y)
    # cv.imshow('Sobel Edges', sobel_edges)
    # cv.imshow('Canny Edges', canny_edges)
    cv.imshow('Thresholded Image', img_thresholded)

    window_names = [
        # 'Original Image',
        'Thresholded Image',
    ]

    # 侦听键盘事件
    is_all_windows_closed = False
    while not is_all_windows_closed:
        key_event = 0xff & cv.waitKey(0)
        if key_event == ord('q'):
            cv.destroyAllWindows()
            exit()
        else:
            print(f'Unexpected key: {key_event}')

        is_all_windows_closed = all(
            cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1
            for window_name in window_names
        )

