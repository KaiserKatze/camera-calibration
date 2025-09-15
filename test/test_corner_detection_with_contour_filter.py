#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from tkinter import filedialog
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN  # 用于聚类分析


def askopenimagefilename():
    return filedialog.askopenfilename(
        title='select an image file to open',
        initialdir='./calibration_images',
        filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp;*.tiff'), ('All files', '*.*')],
    )


def plot_histogram_of_area_of_contours(list_of_contours, min_area=0, max_area=1000):
    area_of_contours = [
        area
        for contour in list_of_contours
        if (area := cv.contourArea(contour)) > min_area
            and area < max_area
    ]
    count_of_contours = len(area_of_contours)
    # max_area_of_contours = max(area_of_contours) if area_of_contours else 0
    # print('Max contour area:', max_area_of_contours, type(max_area_of_contours))
    bin_size = 10
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(
        area_of_contours,
        # bins=int(max_area_of_contours) // bin_size + 1,
        bins=(max_area - min_area) // bin_size + 1,
        color='skyblue',
        edgecolor='black',
        alpha=.7,
    )
    plt.xlabel('Contour Area')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Contour Areas (Total: {count_of_contours} contours)')
    plt.show()
    exit()


def merge_corners_dbscan(corners, eps=3.0, min_samples=1):
    """
    使用 DBSCAN 聚类合并角点
    输入：角点坐标数组，形状为 (N, 2)
    输出：合并后的角点坐标数组
    """
    if len(corners) == 0:
        return corners

    # 执行 DBSCAN 聚类
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(corners)
    labels = db.labels_

    merged_corners = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:  # 噪声点，通常不合并，但你可以选择保留或处理
            continue
        # 找到同一簇的所有角点
        cluster_points = corners[labels == label]
        # 计算簇的质心
        centroid = np.mean(cluster_points, axis=0)
        merged_corners.append(centroid)

    # 处理噪声点（可选：保留或丢弃）
    noise_points = corners[labels == -1]
    if len(noise_points) > 0:
        # 可以选择保留噪声点作为独立角点
        merged_corners.extend(noise_points)

    return np.array(merged_corners, dtype=np.float32)


def get_contour_aspect_ratio(contour):
    _, _, w, h = cv.boundingRect(contour)
    # 计算长宽比（宽度/高度）
    aspect_ratio = w / h
    return aspect_ratio


def detect_and_filter_corners():
    path_image = askopenimagefilename()
    print(f'Opening file {path_image!r} ...')

    if not path_image:
        return exit()

    # 读取图像
    img_original = cv.imread(path_image)
    if img_original is None:
        print('Failed to load image.')
        return exit()
    # 将图像转换为灰度图像
    img_grayscale = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    # 阈值分割
    _, img_thresholded = cv.threshold(img_grayscale, 80, 255, cv.THRESH_BINARY_INV)
    # cv.imshow('Thresholded', img_thresholded)
    # 2. 查找轮廓（以连接角点形成的区域）
    contours, _ = cv.findContours(img_thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv.findContours(img_thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    img_contours_all = img_original.copy()
    cv.drawContours(
        img_contours_all,  # 输出结果
        contours,  # 轮廓列表
        -1,  # 当 `contourIdx` 是负数时，描绘所有轮廓
        (0, 255, 0),  # 绿色 RGB
        2  # 线宽
    )
    cv.imshow('All Contours', img_contours_all)

    # plot_histogram_of_area_of_contours(contours)

    # 3. 计算每个轮廓的面积
    contour_areas = []
    valid_contours = []  # 存储有效的轮廓本身

    for cnt in contours:
        area = cv.contourArea(cnt)
        if 100 < area < 1000:
            contour_areas.append(area)
            valid_contours.append(cnt)

    if not contour_areas:
        print('No contours found.')
        return

    # 将面积列表转换为numpy数组以供聚类（需要reshape）
    X = np.array(contour_areas).reshape(-1, 1)

    # 4. 使用DBSCAN聚类分析面积
    # eps: 领域半径，认为面积相差300像素以内的可能属于同一簇
    # min_samples: 最小簇大小，这里设为1表示允许孤立的点（小簇）
    db = DBSCAN(eps=300, min_samples=1).fit(X)
    labels = db.labels_

    # 5. 找出最大的簇（即“大多数区域”）
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True) # 忽略噪声点（label=-1）
    if len(unique_labels) == 0:
        print('Clustering failed to find any clusters. Using all contours.')
        main_cluster_label = -1  # 特殊情况处理
    else:
        main_cluster_label = unique_labels[np.argmax(counts)]
        print(f'Main cluster label: {main_cluster_label}, size: {np.max(counts)}')

    # 6. 确定主簇的面积范围 [min_area, max_area]
    main_areas = X[labels == main_cluster_label]
    if len(main_areas) > 0:
        area_min = np.min(main_areas)
        area_max = np.max(main_areas)
        print(f'Main area range: [{area_min:.2f}, {area_max:.2f}]')
    else:
        area_min, area_max = 0, np.inf
        print('No main area range found, will keep all contours.')

    # 7. 过滤轮廓：只保留面积在主簇范围内的轮廓
    contours_filtered = []
    for i, cnt in enumerate(valid_contours):
        current_area = contour_areas[i]
        if current_area >= area_min and current_area <= area_max:
            contours_filtered.append(cnt)

    print(f'Filtered contours: {len(contours_filtered)} / {len(valid_contours)} kept.')

    # 基于长宽比，进一步筛选轮廓
    contours_filtered = [
        cnt for cnt in contours_filtered
        if .5 <= get_contour_aspect_ratio(cnt) <= 1.5
    ]

    print(f'After aspect ratio filtering: {len(contours_filtered)} / {len(contours_filtered)} kept.')

    # 如果筛选后无轮廓，退出
    if len(contours_filtered) == 0:
        print("No contours left after aspect ratio filtering.")

    # 8. 绘制过滤后的角点区域（用绿色轮廓表示）
    # 首先创建一个空白图像用于绘制轮廓
    img_contours_filtered = img_original.copy()
    cv.drawContours(img_contours_filtered, contours_filtered, -1, (0, 255, 0), 2)  # 绿色，线宽为2

    # 显示结果
    cv.imshow('Filtered Contours (Green)', img_contours_filtered)

    img_contours_filtered_filled = np.zeros_like(img_grayscale, dtype=np.uint8)
    cv.drawContours(img_contours_filtered_filled, contours_filtered, -1, 255, cv.FILLED)
    # 使用高斯模糊对图像进行预处理，让它变得更平滑，减少噪声干扰
    img_blurred = cv.GaussianBlur(img_contours_filtered_filled, (5, 5), 1.5)
    # 归一化
    img_normalized = cv.normalize(img_blurred, None, 0, 255, cv.NORM_MINMAX)
    # cv.imshow('Filtered Contours (Thresholded)', img_normalized)

    img_float32 = np.float32(img_normalized)
    cv.imshow('Normalized float', img_float32)

    # 1. 使用Harris角点检测
    dst = cv.cornerHarris(
        img_float32,
        blockSize=5,  # 处理角点时考虑的邻域大小
        ksize=3,
        k=0.04,
    )
    dst = cv.dilate(dst, None)

    # 创建角点二值图像：角点处为255，其他为0
    corner_threshold = .01 * dst.max()
    # 非极大值抑制（NMS）
    nms_size = 3  # NMS 的邻域大小
    nms_kernel = cv.getStructuringElement(cv.MORPH_RECT, (nms_size, nms_size))  # 创建一个内核，用来寻找局部最大值
    nms_local_max = cv.dilate(dst, nms_kernel)  # 与原始 dst 进行比较，只保留局部最大值对应的点
    nms_mask = (dst == nms_local_max) & (dst > corner_threshold)

    # # 同时保留原始角点标记（红色）用于对比
    # img_show_corners = img_original.copy()
    # img_show_corners[dst > corner_threshold] = [0, 0, 255]
    # cv.imshow('Original Corners (Red)', img_show_corners)

    # 提取角点坐标：找到所有大于阈值的像素位置
    corner_coordinates = np.column_stack(
        np.where(
            # 这里用 nms_mask.T 是因为 np.where 返回 (y, x)，转置后得到 (x, y) 的格式，便于后续处理
            nms_mask.T
        )
    )
    # corner_coordinates 现在是一个数组，每一行是一个角点的 (y, x) 坐标

    # 2. 转换坐标格式（可选，但更直观）
    # 将 (y, x) 转换为更常用的 (x, y) 点列表
    corner_points = []
    for pt in corner_coordinates:
        x, y = pt[0], pt[1]  # 交换坐标顺序
        corner_points.append([x, y])

    # 转换为NumPy数组以便后续处理
    corner_points = np.array(corner_points, dtype=np.float32)

    # 3. (可选但推荐) 亚像素角点精确化
    # 定义终止条件：最大迭代30次或精度达到0.01
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    # 执行亚像素级角点精确化
    corners_refined = cv.cornerSubPix(img_float32, corner_points, (5,5), (-1,-1), criteria)
    corners_merged = merge_corners_dbscan(corners_refined.reshape(-1, 2), eps=3.0, min_samples=1)

    # 4. 打印或保存角点坐标
    print(f"检测到 {len(corners_merged)} 个角点")
    for i, corner in enumerate(corners_merged):
        print(f"角点 {i+1}: x={corner[0]:.2f}, y={corner[1]:.2f}")

    # 5. 在图像上标记精确化后的角点（例如用蓝色圆圈）
    img_with_refined_corners = img_original.copy()
    for corner in corners_merged:
        x, y = corner.ravel()
        cv.circle(img_with_refined_corners, (int(x), int(y)), 2, (0, 0, 255), -1)  # 蓝色实心圆

    cv.imshow('Refined Corners (Red)', img_with_refined_corners)

if __name__ == '__main__':
    # 打印 python 版本、opencv 版本、numpy 版本、matplotlib 版本
    print(f'Python version: {sys.version}')
    print(f'\tGIL enabled: {sys._is_gil_enabled()}')
    print(f'OpenCV version: {cv.__version__}')
    print(f'Numpy version: {np.__version__}')
    print(f'Matplotlib version: {matplotlib.__version__}')

    # 检测并过滤角点
    detect_and_filter_corners()

    while True:
        key_event = 0xff & cv.waitKey(0)
        if key_event == ord('q'):
            cv.destroyAllWindows()
            exit()
        elif key_event == ord('o'):
            detect_and_filter_corners() # 调用新函数
