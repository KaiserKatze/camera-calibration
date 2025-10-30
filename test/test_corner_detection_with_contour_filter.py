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


# ----------------------- 新增：基于质心的邻域/网格密排筛选 -----------------------
def _contour_centroid(contour):
    """
    计算轮廓的几何中心（重心）。若 moments['m00'] == 0，退化到 boundingRect center。
    返回 (x, y) 浮点坐标。
    """
    M = cv.moments(contour)
    if abs(M.get('m00', 0)) > 1e-12:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        x, y, w, h = cv.boundingRect(contour)
        cx = x + w / 2.0
        cy = y + h / 2.0
    return np.array([cx, cy], dtype=np.float32)


def filter_contours_by_centroid_proximity(contours, eps_multiplier=1.5, min_neighbors=2):
    """
    新增的轮廓筛选：基于轮廓质心的邻域密度与簇聚合来筛除孤立或不按网格排列的轮廓。
    策略：
      1. 计算每个轮廓的质心，得到 Nx2 数组。
      2. 计算每个点的最近邻距离，并取中位数 median_nn 作为参考网格间距。
      3. 用 DBSCAN(eps = eps_multiplier * median_nn) 找到主簇（最大簇），先保留主簇内点。
      4. 对于主簇内的点，再计算局部邻居数（半径 = 1.2*median_nn），要求邻居至少为 min_neighbors（排除自身）。
      5. 返回满足条件的轮廓列表（顺序与输入保持一致）。
    参数说明：
      - eps_multiplier: 用于 DBSCAN eps 的倍数（默认 1.5，越大更宽松）
      - min_neighbors: 局部邻居最少数量阈值（默认 2，表示需要至少两个近邻）
    """
    n = len(contours)
    if n == 0:
        return contours

    # 计算质心数组
    centroids = np.array([_contour_centroid(cnt) for cnt in contours], dtype=np.float32)  # shape (n,2)

    # 如果数据点太少，直接返回（避免无法估计网格间距）
    if n <= min_neighbors:
        return contours

    # 计算两两距离矩阵（留意不要太大，通常轮廓数量不会非常大）
    diffs = centroids[:, None, :] - centroids[None, :, :]  # shape (n,n,2)
    dists = np.linalg.norm(diffs, axis=2)  # shape (n,n)

    # 对角线是 0（自己），找到每个点的最近非零邻居距离
    # 为稳健，若某行除了自己全部为零（不太可能）则用很小值避免异常
    nn_dists = []
    for i in range(n):
        row = dists[i]
        # 排除自身距离为0
        nonzero = row[row > 1e-12]
        if nonzero.size == 0:
            nn = 1.0  # 退化情况
        else:
            nn = np.min(nonzero)
        nn_dists.append(nn)
    nn_dists = np.array(nn_dists, dtype=np.float32)

    # 参考网格间距：取中位数，稳健对抗异常点
    median_nn = float(np.median(nn_dists))
    if median_nn <= 1e-6:
        # 如果中位数过小，说明点几乎重合或数据异常，这种情况下只保留全部以避免误筛
        return contours

    # 使用 DBSCAN 找到主簇（密排的一组质心）
    db_eps = eps_multiplier * median_nn
    db = DBSCAN(eps=db_eps, min_samples=1).fit(centroids)
    labels = db.labels_  # -1 表示噪声
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        # 没找到主簇，退化为保留全部
        return contours

    main_label = unique_labels[np.argmax(counts)]
    in_main_cluster = (labels == main_label)

    # 局部邻居计数（在 radius = 1.2 * median_nn 范围内）
    local_radius = 1.2 * median_nn
    neighbor_counts = (dists <= local_radius).sum(axis=1) - 1  # 减去自身
    locally_dense = neighbor_counts >= min_neighbors

    # 最终条件：既属于主簇，又局部密度足够
    keep_mask = in_main_cluster & locally_dense

    # 如果筛掉过多（例如主簇数量非常少导致全部被剔除），则退化到只用主簇判断
    if keep_mask.sum() == 0:
        keep_mask = in_main_cluster

    # 构建返回的轮廓列表（保持原输入顺序）
    filtered = [cnt for keep, cnt in zip(keep_mask, contours) if keep]
    return filtered
# ----------------------- 新增结束 -----------------------


# ----------------------- 新增：状态栏工具 -----------------------
def ensure_bgr(img):
    """确保图像是 BGR 3 通道，用于在其上绘制状态栏与文字。"""
    if img is None:
        return None
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    return img.copy()


def show_image_with_statusbar(window_name, image, bar_height=24):
    """
    在窗口底部增加状态栏，并在鼠标位于窗口内时显示坐标。
    注意：需要在主循环中用 waitKey(>0) 轮询，以处理鼠标事件和刷新。
    """
    img_bgr = ensure_bgr(image)
    h, w = img_bgr.shape[:2]
    canvas = np.zeros((h + bar_height, w, 3), dtype=np.uint8)

    font = cv.FONT_HERSHEY_SIMPLEX
    baseline_offset = 6  # 文字距离条带底部的像素

    def render(text: str):
        # 复制原图到画布上方
        canvas[:h] = img_bgr
        # 画状态条背景
        cv.rectangle(canvas, (0, h), (w, h + bar_height), (32, 32, 32), thickness=-1)
        # 文字
        if text:
            cv.putText(
                canvas, text,
                (5, h + bar_height - baseline_offset),
                font, 0.5, (255, 255, 255), 1, cv.LINE_AA
            )
        cv.imshow(window_name, canvas)

    def on_mouse(event, x, y, flags, param):
        # 仅当鼠标在图像区域（不包括状态栏条带）内时显示坐标
        if 0 <= x < w and 0 <= y < h:
            render(f'x={x}, y={y}')
        else:
            # 鼠标不在图像区域（可能在状态栏或窗口外），清空提示
            render('')

    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(window_name, on_mouse)
    render('Move mouse to see coordinates')
# ----------------------- 新增：状态栏工具 -----------------------


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
        if area_min <= current_area <= area_max:
            contours_filtered.append(cnt)
    print(f'Filtered contours: {len(contours_filtered)} / {len(valid_contours)} kept.')

    # 基于长宽比进一步筛选
    contours_filtered = [
        cnt for cnt in contours_filtered
        if .5 <= get_contour_aspect_ratio(cnt) <= 1.5
    ]

    print(f'After aspect ratio filtering: {len(contours_filtered)} / {len(contours_filtered)} kept.')

    # 如果筛选后无轮廓，退出
    if len(contours_filtered) == 0:
        print("No contours left after aspect ratio filtering.")

    # 通过质心的局部密度与主簇判断，去除孤立或未紧密排列的轮廓
    contours_before_centroid_filter = len(contours_filtered)
    contours_filtered = filter_contours_by_centroid_proximity(
        contours_filtered,
        eps_multiplier=1.5,   # DBSCAN eps = 1.5 * median_nn（可调整）
        min_neighbors=2       # 局部邻居至少 2 个（可调整）
    )
    print(f'After centroid-proximity filtering: {len(contours_filtered)} / {contours_before_centroid_filter} kept.')

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
    corners_merged = corners_merged[np.lexsort((corners_merged[:, 0], corners_merged[:, 1]))]

    # 4. 打印或保存角点坐标
    print(f"检测到 {len(corners_merged)} 个角点")
    for i, corner in enumerate(corners_merged):
        print(f"角点 {i+1}: x={corner[0]:.2f}, y={corner[1]:.2f}")

    # 5. 在图像上标记精确化后的角点（例如用蓝色圆圈）
    img_with_refined_corners = img_original.copy()
    for corner in corners_merged:
        x, y = corner.ravel()
        cv.circle(img_with_refined_corners, (int(x), int(y)), 2, (0, 0, 255), -1)  # 蓝色实心圆

    show_image_with_statusbar('Refined Corners (Red)', img_with_refined_corners)


if __name__ == '__main__':
    # 打印 python 版本、opencv 版本、numpy 版本、matplotlib 版本
    print(f'Python version: {sys.version}')
    print(f'\tGIL enabled: {sys._is_gil_enabled()}')
    print(f'OpenCV version: {cv.__version__}')
    print(f'Numpy version: {np.__version__}')
    print(f'Matplotlib version: {matplotlib.__version__}')

    # 检测并过滤角点
    detect_and_filter_corners()

    # 主循环：用小延时轮询（而不是 waitKey(0)）以处理鼠标事件并刷新状态栏
    while True:
        key_event = 0xff & cv.waitKey(20)  # 20ms 轮询
        if key_event == ord('q'):
            cv.destroyAllWindows()
            exit()
        elif key_event == ord('o'):
            detect_and_filter_corners()  # 重新选择并处理新图像
