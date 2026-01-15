#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import typing

import numpy as np
import scipy.optimize
import scipy


np.set_printoptions(linewidth=np.inf)

DEBUG_MODE = False

#================================================================================================
#
# 相机标定
#
#================================================================================================


class RadialCalibrator:
    @classmethod
    def make_matrix_L(cls, model: np.ndarray, pixel: np.ndarray):
        model_h, model_w = model.shape
        u = pixel[:, 0]
        v = pixel[:, 1]
        vm = v[:, None] * model
        um = -u[:, None] * model
        L = np.concatenate([vm, um], axis=1)
        assert L.shape == (model_h, 2 * model_w)
        return L

    @classmethod
    def guess_projective_matrix(cls, model, pixel):
        L = cls.make_matrix_L(model, pixel)
        # print(f'{L=}')
        _, _, V = np.linalg.svd(L, full_matrices=True, compute_uv=True, hermitian=False)
        m = V[:, -1]
        assert len(m) == 2 * 4  # 可以估计的只有投影矩阵的两个行向量
        return m

    @classmethod
    def infer_intrinsic_params(cls, model: np.ndarray, pixel: np.ndarray):
        model_h, model_w = model.shape
        if model_w == 2:  # 如果用户只传入了模型的非齐次仿射坐标，则自动转换为齐次射影坐标
            model = np.block([model, np.zeros((model_h, 1)), np.ones((model_h, 1))])
        model_h, model_w = model.shape
        pixel_h, pixel_w = pixel.shape
        assert model_h == pixel_h, f'模型点、像点数量不相等: 模型点数量={model_h}, 像点数量={pixel_h}'
        assert pixel_h > 6, f'样本点数量太少: {pixel_h}'
        assert model_w == 4, '必须输入模型的齐次坐标'
        assert pixel_w == 2, '必须输入影像的非齐次坐标'


        print(f'model={model[:5]}')
        print(f'pixel={pixel[:5]}')

        # 求解系统的线性部分，找出近似解
        # 使用该解作为系统的初始条件
        m1_and_m2 = cls.guess_projective_matrix(model, pixel)
        # 猜测 m3 = (1,1,1,1)
        m3 = np.array([0, 0, 1, 0], dtype=np.float64) + 1e-4 * np.random.randn(4)
        # 构造优化向量
        x0 = np.concatenate([m1_and_m2, m3, np.zeros(3)])
        # 这里 4 表示投影矩阵的行向量 m3 的维数，3 表示畸变因子个数（k1,k2,k3）
        print(f'{x0=}')
        print(f'闭式估计得到的投影矩阵：\n Mx={x0[:3*4].reshape(3,4)}')

        # 用牛顿法
        def residuals(x) -> np.float64:
            nr, nc = 3, 4
            Mx = x[:nr * nc].reshape(nr, nc)  # 取出投影矩阵
            k1, k2, k3 = x[nr * nc:]  # 取出径向畸变系数

            # ---------- 尺度规一化（避免 M 的整体尺度导致病态） ----------
            denom = Mx[2, 2]  # 取出投影矩阵 Mx 的 (3,3) 元素
            if abs(denom) > 1e-12:
                Mx = Mx / denom
            else:
                # 若 Mx[2,2] 过小，用 Frobenius 范数做规一化
                fn = np.linalg.norm(Mx)
                if fn < 1e-12:
                    fn = 1.0
                Mx = Mx / fn

            # 投影到像平面（齐次）
            pixel_homo = model @ Mx.T  # 像素坐标系中的齐次坐标
            if DEBUG_MODE:
                print(f'pixel_homo={pixel_homo[:5]}')

            # 计算像素坐标系中的非齐次坐标
            pixel_w = pixel_homo[:, 2]
            # 避免除以零
            pixel_w_safe = np.where(np.abs(pixel_w) < 1e-12, np.sign(pixel_w) * 1e-12, pixel_w)
            pixel_x = pixel_homo[:, 0] / pixel_w_safe
            pixel_y = pixel_homo[:, 1] / pixel_w_safe

            # 计算径向畸变
            r_squared = pixel_x * pixel_x + pixel_y * pixel_y
            one_over_lambda = 1. / (1. + k1 * r_squared + k2 * r_squared ** 2 + k3 * r_squared ** 3)

            # 施加径向畸变
            pixel_x = one_over_lambda * pixel_x
            pixel_y = one_over_lambda * pixel_y

            # 计算残差向量
            residue = np.empty(2 * model_h, dtype=np.float64)
            residue[:model_h] = pixel_x - pixel[:, 0]
            residue[model_h:] = pixel_y - pixel[:, 1]

            return residue

        # 利用牛顿法逼近
        try:
            optimize_result = scipy.optimize.least_squares(
                residuals,
                x0=x0,
                method='lm',  # Levenberg-Marquardt algorithm
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
                max_nfev=5000,
                verbose=2,
            )
        except ValueError as e:  # ValueError: Residuals are not finite in the initial point.
            if e.args[0] == 'Residuals are not finite in the initial point.':
                global DEBUG_MODE
                DEBUG_MODE = True
                err_residue = residuals(x0)
                print(f'{err_residue=}')
            exit(-1)
        print(f'{optimize_result=}')
        x_opt = optimize_result.x
        M_opt = x_opt[:12].reshape(3, 4)
        k_opt = x_opt[12:]
        print('Estimated M (unnormalized):\n', M_opt)
        print('Estimated distortion k1,k2,k3:', k_opt)

        return {
            'M': M_opt,
            'k': k_opt,
        }



#================================================================================================
#
# 图像处理
#
#================================================================================================





import sys
import glob

import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN  # 用于聚类分析


m = 8  # 网格行数
n = 8  # 网格列数
a = 25  # 每个黑色正方形的边长（像素）
b = 30  # 相邻正方形之间的间距（像素）
c = 50  # 正方形与图片边缘的最小距离（像素）



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
                (x_start, y_start),
                (x_start, y_end),
                (x_end, y_start),
                (x_end, y_end),
            ])
    coordinates = np.array(coordinates, dtype=np.float32)
    # 行优先排序（先 y 后 x）
    idx = np.lexsort((coordinates[:,0], coordinates[:,1]))
    return coordinates[idx]


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

    for label in np.unique(labels):
        if label == -1:  # 噪声点，通常不合并，但你可以选择保留或处理
            continue
        # 找到同一簇的所有角点
        cluster_points = corners[labels == label]
        # 计算簇的质心
        centroid = np.mean(cluster_points, axis=0)
        merged_corners.append(centroid)

    merged_corners = np.array(merged_corners, dtype=np.float32)
    # 关键：全局按 y 再按 x 排序，得到行优先顺序
    idx = np.lexsort((merged_corners[:,0], merged_corners[:,1]))  # 主键 y，次键 x
    return merged_corners[idx]


def sort_corners_grid(corners, n_rows, n_cols):
    """
    对角点进行行优先排序
    输入：
        corners: (N,2) numpy array
        n_rows: 棋盘格行数
        n_cols: 棋盘格列数
    输出：
        corners_sorted: (N,2) numpy array, 按行优先排序
    """
    # 按 y 坐标排序得到行
    idx_sorted_by_y = np.argsort(corners[:,1])
    corners_sorted_by_y = corners[idx_sorted_by_y]

    # 将角点划分为 n_rows 行，每行 n_cols 点
    rows = np.array_split(corners_sorted_by_y, n_rows)

    corners_sorted = []
    for row in rows:
        # 每行按 x 坐标排序
        row_sorted = row[np.argsort(row[:,0])]
        corners_sorted.append(row_sorted)
    corners_sorted = np.vstack(corners_sorted)
    return corners_sorted


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
    corners_merged = corners_merged[np.lexsort((corners_merged[:, 0], corners_merged[:, 1]))]

    if len(corners_merged) != m * n * 4:  # 检查角点数量与预期是否一致
        raise AssertionError(f"角点数量 {len(corners_merged)} 不符合预期 {m*n*4}")

    image_shape = img_grayscale.shape[::-1]

    corners_merged_sorted = sort_corners_grid(corners_merged, m, n*4)  # m行, 每行 n*4 点

    return corners_merged_sorted, image_shape


#================================================================================================
#
# 数据生成
#
#================================================================================================




def make_model_and_pixel():
    path_glob = 'calibration_images/*.jpg'

    # 获取所有标定图像的路径
    images = glob.glob(path_glob)  # 修改为你的图像路径
    imgpoints = []  # 像点在相机坐标系中的坐标
    for image in images:
        corners, _ = detect_corners(image)
        if corners is None:
            print(f"未能在图像 {image} 中检测到角点，跳过该图像!")
            continue
        imgpoints.append(corners)

    # 准备模型点（用世界坐标表示）
    object_points_in_world_coordinates = generate_object_points_in_world_coordinates()
    objpoints = [object_points_in_world_coordinates] * len(imgpoints)
    objpoints = np.concatenate(objpoints, axis=0)
    imgpoints = np.concatenate(imgpoints, axis=0)
    return objpoints, imgpoints


def save_npy(path: str, arr: np.ndarray) -> None:
    """保存单个数组为 .npy"""
    np.save(path, arr, allow_pickle=False)  # 不用 pickle 更安全
    # note: if path has no extension, numpy will add .npy


def load_npy(path: str) -> np.ndarray:
    """从 .npy 加载单个数组"""
    return np.load(path, allow_pickle=False)


def save_mat(path: str, mat: np.ndarray) -> None:
    scipy.io.savemat(path, {'data': mat})



if __name__ == '__main__':
    model, pixel = make_model_and_pixel()
    save_npy('model.npy', model)
    save_npy('pixel.npy', pixel)
    save_mat('model.mat', model)
    save_mat('pixel.mat', pixel)

    # print(f'{model.shape=}')
    # print(f'{pixel.shape=}')
    # print(f'model={model[:5]}')
    # print(f'pixel={pixel[:5]}')

    model = load_npy('model.npy')
    pixel = load_npy('pixel.npy')
    RadialCalibrator.infer_intrinsic_params(model, pixel)
