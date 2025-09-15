#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import glob

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import DBSCAN  # 用于聚类分析

from generate_model_plane import m, n, a, b, c

def generate_object_points_in_world_coordinates():
    """
    生成模型平面
    """
    coordinates = []
    for i in range(m):
        for j in range(n):
            # 计算当前正方形的左上角坐标
            x_start = c + j * (a + b)
            y_start = c + i * (a + b)

            # 计算当前正方形的右下角坐标
            x_end = x_start + a
            y_end = y_start + a

            coordinates.extend([
                (x_start, y_start, 0),
                (x_start, y_end, 0),
                (x_end, y_start, 0),
                (x_end, y_end, 0),
            ])
    return np.array(coordinates, dtype=np.float32)


def get_contour_aspect_ratio(contour: cv.typing.MatLike):
    """
    给定轮廓，计算长宽比（宽度/高度）
    """
    _, _, w, h = cv.boundingRect(contour)
    aspect_ratio = w / h
    return aspect_ratio


def filter_contours_by_area_and_aspect_ratio(contours, min_area=100, max_area=1000):
    contour_areas = []  # 存储面积符合要求的轮廓面积
    valid_contours = []  # 存储面积符合要求的轮廓
    for contour in contours:  # 首先对轮廓按照面积初步过滤
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            contour_areas.append(area)
            valid_contours.append(contour)

    if not contour_areas:
        print('No contours found.')
        return

    # 将面积列表转换为numpy数组以供聚类（需要reshape）
    X = np.array(contour_areas).reshape(-1, 1)

    # 使用DBSCAN聚类分析面积
    #   eps: 领域半径，认为面积相差300像素以内的可能属于同一簇
    #   min_samples: 最小簇大小，这里设为1表示允许孤立的点（小簇）
    db = DBSCAN(eps=300, min_samples=1).fit(X)
    labels = db.labels_

    # 找出最大的簇（即“大多数区域”）
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True) # 忽略噪声点（label=-1）
    if len(unique_labels) == 0:
        print('Clustering failed to find any clusters. Using all contours.')
        main_cluster_label = -1  # 特殊情况处理
    else:
        main_cluster_label = unique_labels[np.argmax(counts)]
        # print(f'Main cluster label: {main_cluster_label}, size: {np.max(counts)}')

    # 确定主簇的面积范围 [min_area, max_area]
    main_areas = X[labels == main_cluster_label]
    if len(main_areas) > 0:
        area_min = np.min(main_areas)
        area_max = np.max(main_areas)
        # print(f'Main area range: [{area_min:.2f}, {area_max:.2f}]')
    else:
        area_min, area_max = 0, np.inf
        print('No main area range found, will keep all contours.')

    # 过滤轮廓：只保留面积在主簇范围内的轮廓
    contours_filtered = []
    for i, cnt in enumerate(valid_contours):
        current_area = contour_areas[i]
        if current_area >= area_min and current_area <= area_max:
            contours_filtered.append(cnt)

    # print(f'Filtered contours: {len(contours_filtered)} / {len(valid_contours)} kept.')

    # 基于长宽比，进一步筛选轮廓
    contours_filtered = [
        cnt for cnt in contours_filtered
        if .5 <= get_contour_aspect_ratio(cnt) <= 1.5
    ]

    # print(f'After aspect ratio filtering: {len(contours_filtered)} / {len(contours_filtered)} kept.')

    # 如果筛选后无轮廓，退出
    if len(contours_filtered) == 0:
        print("No contours left after aspect ratio filtering.")

    return contours_filtered


def merge_corners_dbscan(corners: np.ndarray, eps: float = 3.0, min_samples: int = 1):
    """
    使用 DBSCAN 聚类，合并角点
    输入：
        @param `corners`: 角点坐标数组，形状为 (N, 2)
        @param `eps`: DBSCAN 的 eps 参数，表示两个样本被看作是邻居的最大距离
        @param `min_samples`: DBSCAN 的 min_samples 参数，表示一个簇的最小样本数
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


def detect_corners(path_image):
    if not path_image:
        print('No image path provided.')
        return

    # 读取图像
    img_original = cv.imread(path_image)
    if img_original is None:
        print('Failed to load image.')
        return

    # 将图像转换为灰度图像
    img_grayscale = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    # 阈值分割
    _, img_thresholded = cv.threshold(img_grayscale, 80, 255, cv.THRESH_BINARY_INV)
    # 查找轮廓（以连接角点形成的区域）
    contours, _ = cv.findContours(img_thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 计算每个轮廓的面积
    contours_filtered = filter_contours_by_area_and_aspect_ratio(contours)
    # 首先创建一个空白图像用于绘制轮廓
    img_contours_filtered_filled = np.zeros_like(img_grayscale, dtype=np.uint8)
    cv.drawContours(img_contours_filtered_filled, contours_filtered, -1, 255, cv.FILLED)
    # 使用高斯模糊对图像进行预处理，让它变得更平滑，减少噪声干扰
    img_blurred = cv.GaussianBlur(img_contours_filtered_filled, (5, 5), 1.5)
    # 归一化
    img_normalized = cv.normalize(img_blurred, None, 0, 255, cv.NORM_MINMAX)
    img_float32 = np.float32(img_normalized)
    # 使用Harris角点检测
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
    # 提取角点坐标：找到所有大于阈值的像素位置
    corner_coordinates = np.column_stack(
        np.where(
            # 这里用 nms_mask.T 是因为 np.where 返回 (y, x)，转置后得到 (x, y) 的格式，便于后续处理
            nms_mask.T
        )
    )
    # corner_coordinates 现在是一个数组，每一行是一个角点的 (y, x) 坐标
    # 转换坐标格式（可选，但更直观），将 (y, x) 转换为更常用的 (x, y) 点列表
    corner_points = []
    for pt in corner_coordinates:
        x, y = pt[0], pt[1]  # 交换坐标顺序
        corner_points.append([x, y])
    # 转换为NumPy数组以便后续处理
    corner_points = np.array(corner_points, dtype=np.float32)
    # (可选但推荐) 亚像素角点精确化
    # 定义终止条件：最大迭代30次或精度达到0.01
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    # 执行亚像素级角点精确化
    corners_refined = cv.cornerSubPix(img_float32, corner_points, (5,5), (-1,-1), criteria)
    corners_merged = merge_corners_dbscan(corners_refined.reshape(-1, 2), eps=3.0, min_samples=1)

    if len(corners_merged) != m * n * 4:  # 检查角点数量与预期是否一致
        print(f"角点数量 {len(corners_merged)} 不符合预期 {m*n*4}")
        return

    image_shape = img_grayscale.shape[::-1]

    return corners_merged, image_shape


def construct_parameters_for_camera_calibration(path_glob):
    # 获取所有标定图像的路径
    images = glob.glob(path_glob)  # 修改为你的图像路径
    imgpoints = []  # 像点在相机坐标系中的坐标
    image_shape = None
    for image in images:
        corners, image_shape = detect_corners(image)
        if corners is None:
            print(f"未能在图像 {image} 中检测到角点，跳过该图像!")
            continue
        imgpoints.append(corners)

    # 准备模型点（用世界坐标表示）
    object_points_in_world_coordinates = generate_object_points_in_world_coordinates()
    objpoints = [object_points_in_world_coordinates] * len(imgpoints)

    return objpoints, imgpoints, image_shape


if __name__ == '__main__':
    # 打印 python 版本、opencv 版本、numpy 版本、matplotlib 版本
    print(f'Python version: {sys.version}')
    print(f'\tGIL enabled: {sys._is_gil_enabled()}')
    print(f'OpenCV version: {cv.__version__}')
    print(f'Numpy version: {np.__version__}')
    print(f'Matplotlib version: {matplotlib.__version__}')

    objpoints, imgpoints, image_shape = construct_parameters_for_camera_calibration('calibration_images/*.jpg')

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    print("\n=== 标定结果 ===")
    print("相机内参矩阵 (K):\n", camera_matrix)
    print("\n畸变系数 (k1, k2, p1, p2, k3...):\n", dist_coeffs.ravel())

    # 计算重投影误差以评估标定精度
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        # error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        imgpoints2_reshaped = imgpoints2.reshape(-1, 2)  # 将 imgpoints2 的形状从 (N,1,2) 重塑为 (N,2)
        diff = imgpoints[i] - imgpoints2_reshaped
        error = np.linalg.norm(diff, ord=2) / len(imgpoints2_reshaped)  # 计算 L2 范数并归一化
        mean_error += error

    print(f"\n平均重投影误差: {mean_error/len(objpoints):.5f} 像素")

    # 保存标定参数
    np.savez('camera_calibration_params.npz',
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
            reprojection_error=mean_error/len(objpoints))
    print("\n参数已保存至 'camera_calibration_params.npz'")
