#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import glob
from tkinter import filedialog

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import DBSCAN  # 用于聚类分析


def askopenimagefilename():
    return filedialog.askopenfilename(
        title='select an image file to open',
        initialdir='./calibration_images',
        filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp;*.tiff'), ('All files', '*.*')],
    )


def correct_distortion():
    path_image = askopenimagefilename()
    print(f'Opening file {path_image!r} ...')

    if not path_image:
        return

    # 读取图像
    img_original = cv.imread(path_image)
    if img_original is None:
        print('Failed to load image.')
        return

    data: np.lib.npyio.NpzFile = np.load('camera_calibration_params.npz')
    print(f'{type(data)=}')
    print('data keys=', list(data.keys()))
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    # rvecs = data['rvecs']
    # tvecs = data['tvecs']
    reprojection_error = data['reprojection_error']
    print("\n=== 已加载标定参数 ===")
    print("相机内参矩阵 (K):\n", camera_matrix)
    print("\n畸变系数 (k1, k2, p1, p2, k3...):\n", dist_coeffs.ravel())
    print(f"\n平均重投影误差: {reprojection_error:.5f} 像素")

    h, w = img_original.shape[:2]

    # 获取优化后的新相机矩阵和ROI区域
    img_size = (w, h)
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        img_size,
        0,
        img_size
    )

    # 校正图像
    dst = cv.undistort(img_original, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 裁剪图像（可选，根据ROI）
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # 并排显示原图与校正后的图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    plt.title('Undistorted Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 打印 python 版本、opencv 版本、numpy 版本、matplotlib 版本
    print(f'Python version: {sys.version}')
    print(f'\tGIL enabled: {sys._is_gil_enabled()}')
    print(f'OpenCV version: {cv.__version__}')
    print(f'Numpy version: {np.__version__}')
    print(f'Matplotlib version: {matplotlib.__version__}')

    # 使用标定参数校正图像畸变
    correct_distortion()
