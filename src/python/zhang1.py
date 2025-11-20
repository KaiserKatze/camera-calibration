#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import functools
import typing

import numpy as np
import scipy.optimize
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


np.set_printoptions(linewidth=np.inf)
svd = functools.partial(np.linalg.svd, full_matrices=True, compute_uv=True, hermitian=False)


import logging
import pathlib
import logging.handlers


def getLogger(name: str) -> logging.Logger:
    name = pathlib.Path(name).stem
    dual_stack_logger = logging.getLogger(name)  # 创建一个logger
    dual_stack_logger.setLevel(logging.DEBUG)  # 设置日志级别

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 创建一个formatter，用于定义日志格式
    pathlib.Path('logs').mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        filename=pathlib.Path('logs', f'{name}.log'),
        mode='w',
        encoding='utf-8',
    )  # 创建一个handler，用于将日志写入文件
    file_handler.setLevel(logging.DEBUG)  # 设置handler的日志级别
    file_handler.setFormatter(formatter)
    dual_stack_logger.addHandler(file_handler)  # 将handler添加到logger

    console_handler = logging.StreamHandler()  # 创建一个handler，用于将日志输出到控制台
    console_handler.setLevel(logging.ERROR)  # 设置handler的日志级别为ERROR
    console_handler.setFormatter(formatter)  # 为控制台handler设置formatter
    dual_stack_logger.addHandler(console_handler)  # 将控制台handler添加到logger

    return dual_stack_logger


logger = getLogger(__file__)



def normalize_points(points):
    """
    实现论文 [In defence of the 8-point algorithm Section 6.1](https://ieeexplore.ieee.org/document/466816) 描述的各向同性归一化 (Isotropic Scaling)。

    Args:
        points: 形状为 (N, 2) 的图像点坐标数组。

    Returns:
        new_points: 归一化后的点坐标，形状为 (N, 2)。
        T: 3x3 的变换矩阵，满足 new_points_homogeneous = T * old_points_homogeneous。
    """
    # 1. 计算重心 (Centroid) [cite: 166]
    centroid = np.mean(points, axis=0)
    cx, cy = centroid[0], centroid[1]

    # 将点平移到原点
    shifted_points = points - centroid

    # 2. 计算平均距离
    # 计算每个点到原点的欧几里得距离 [cite: 34]
    distances = np.sqrt(np.sum(shifted_points**2, axis=1))
    mean_dist = np.mean(distances)

    # 3. 计算缩放因子，使得平均距离等于 sqrt(2) [cite: 167]
    scale = np.sqrt(2) / mean_dist

    # 4. 构建变换矩阵 T
    # T = [scale,   0,   -scale * cx]
    #     [  0,   scale, -scale * cy]
    #     [  0,     0,       1      ]
    T = np.array([
        [scale, 0, -scale * cx],
        [0, scale, -scale * cy],
        [0, 0, 1]
    ])

    # 应用变换矩阵得到归一化后的点
    # 首先转换为齐次坐标 (N, 3)
    points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
    # 矩阵乘法: (T @ point.T).T
    normalized_points_homo = (T @ points_homo.T).T

    # 返回非齐次坐标 (N, 2) 和 变换矩阵 T
    return normalized_points_homo[:, :2], T



def generate_model_points():
    """
    生成模型平面。
    一共生成 N = m * n * 4 个点。

    Returns:
        points_2d_homo: 模型点齐次坐标，形状为 (N, 3)。
    """

    m = 8  # 网格行数
    n = 8  # 网格列数
    a = 25  # 每个黑色正方形的边长（像素）
    b = 30  # 相邻正方形之间的间距（像素）
    c = 50  # 正方形与图片边缘的最小距离（像素）

    points_2d = []
    for i in range(m):
        for j in range(n):
            # 计算当前正方形的左上角坐标
            x_start = c + j * (a + b)
            y_start = c + i * (a + b)

            # 计算当前正方形的右下角坐标
            x_end = x_start + a
            y_end = y_start + a

            points_2d.extend([
                (x_start, y_start),
                (x_start, y_end),
                (x_end, y_start),
                (x_end, y_end),
            ])
    points_2d = np.array(points_2d, dtype=np.float32)

    # 居中
    x_min, y_min = np.min(points_2d, axis=0)
    x_max, y_max = np.max(points_2d, axis=0)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    points_2d -= (x_center, y_center)

    # 行优先排序（先 y 后 x）
    idx = np.lexsort((points_2d[:,0], points_2d[:,1]))
    points_2d = points_2d[idx]

    # 转换为齐次坐标 `(x,y,1)`，第三个分量为 1
    num_points = points_2d.shape[0]
    ones = np.ones((num_points, 1))
    points_2d_homo = np.hstack((points_2d, ones))
    assert points_2d_homo.shape == (m * n * 4, 3)
    return points_2d_homo


class Rotation:
    def __init__(self, Rx: float, Ry: float, Rz: float):
        self.R = scipy.spatial.transform.Rotation.from_euler('xyz', [Rx, Ry, Rz], degrees=True).as_matrix()
        self.Rx = Rx
        self.Ry = Ry
        self.Rz = Rz
        # logger.debug(f'{Rx=}')
        # logger.debug(f'{Ry=}')
        # logger.debug(f'{Rz=}')

    @classmethod
    def randomize(cls):
        Rx = np.random.uniform(-60, 60)
        Ry = np.random.uniform(-60, 60)
        Rz = np.random.uniform(-60, 60)
        return cls(Rx, Ry, Rz)


class Translation:
    def __init__(self, Tx: float, Ty: float, Tz: float):
        self.T = np.array([[Tx], [Ty], [Tz]])
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz
        # logger.debug(f'{Tx=}')
        # logger.debug(f'{Ty=}')
        # logger.debug(f'{Tz=}')

    @classmethod
    def randomize(cls):
        Tx = np.random.uniform(-100, 100)
        Ty = np.random.uniform(-100, 100)
        Tz = np.random.uniform(200, 500)
        return cls(Tx, Ty, Tz)


class CameraModel:
    def __init__(self, d: float, a: float, b: float, theta: float, u0: float, v0: float):
        alpha = d / a
        beta = d / b
        self.K = np.array([
            [alpha, -alpha / np.tan(theta), u0],
            [0, beta / np.sin(theta), v0],
            [0, 0, 1],
        ])
        logger.debug(f'真实的相机内参矩阵 K=\n{self.K}')

    def _arbitrary_project(self, model_2d_homo: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
        """
        将世界坐标系中的模型点投影到像素坐标系，输入模型点的齐次坐标，返回像素点的齐次坐标

        Args:
            model_2d_homo 平面标定板上的模型点的二维齐次坐标 (x,y,1)

        Returns:
            model_2d_homo 模型点的二维齐次坐标 (x,y,1)
            pixel_2d_homo 像素点的二维齐次坐标 (u,v,1)
            H 单应性
        """
        assert model_2d_homo.shape[1] == 3
        logger.debug(f'模型点的二维齐次坐标 model_2d_homo=\n{model_2d_homo[:5]}')
        # 计算单应性 H
        H = self.K @ np.hstack((rotation[:, :2], translation))
        assert H.shape == (3, 3)
        logger.debug(f'真实的旋转矩阵 R=\n{rotation}')
        logger.debug(f'真实的平移向量 T=\n{translation}')
        logger.debug(f'真实的单应性 H=\n{H}')
        # 计算像素点的齐次坐标
        pixel_2d_homo = model_2d_homo @ H.T
        # 计算像素点的非齐次坐标
        pixel_2d = pixel_2d_homo[:, :2] / pixel_2d_homo[:, 2:3]
        # 给像素点的非齐次坐标添加高斯噪声（均值=0，标准差=0.5 像素）
        noise = np.random.normal(0, 0.5, pixel_2d.shape)
        pixel_2d += noise
        # 返回像素点的齐次坐标
        num_points = pixel_2d.shape[0]
        ones = np.ones((num_points, 1))
        pixel_2d_homo = np.hstack((pixel_2d, ones))
        assert pixel_2d_homo.shape[1] == 3
        return model_2d_homo, pixel_2d_homo, rotation, translation, H

    def randomly_project(self, model_2d_homo: np.ndarray):
        # 随机生成旋转矩阵和位移向量
        rotation = Rotation.randomize().R
        translation = Translation.randomize().T
        return self._arbitrary_project(model_2d_homo, rotation, translation)


class ZhangCameraCalibration:
    @classmethod
    def evaluate_relative_error(cls, estimated_homography: np.ndarray, ground_truth_homography: np.ndarray):
        normalized_ground_truth_homography = ground_truth_homography / ground_truth_homography[2,2]
        normalized_estimated_homography = estimated_homography / estimated_homography[2,2]
        relative_error = np.linalg.norm(normalized_ground_truth_homography - normalized_estimated_homography)
        relative_error /= np.linalg.norm(normalized_ground_truth_homography)
        logger.debug(f'单应性相对误差 ={relative_error*100:.6f}%')
        return relative_error

    @classmethod
    def infer_homography_without_radial_distortion(cls, model: np.ndarray, pixel: np.ndarray):
        # 这里构造的矩阵 L 与张正友的报告中的相同
        # L := [
        #     [M1.T, 0, - u1 * M1.T],
        #     [0, M1.T, - v1 * M1.T],
        #     ...,
        #     [Mn.T, 0, - un * Mn.T],
        #     [0, Mn.T, - vn * Mn.T],
        # ]
        assert model.shape[1] == pixel.shape[1] == 3, '必须输入模型点、像素点的二维齐次坐标!'
        assert model.shape[0] == pixel.shape[0] > 6, '样本数量太少!'
        model_h = model.shape[0]
        # 矢量化构造 L （每个对应产生两行）
        zeros = np.zeros_like(model)
        u = pixel[:, 0:1]
        v = pixel[:, 1:2]
        # logger.debug(f'u=\n{u[:5]}')
        # logger.debug(f'v=\n{v[:5]}')
        # logger.debug(f'model=\n{model[:5]}')
        row1 = np.hstack([model, zeros, -u * model])
        assert row1.shape == (model_h, 9)
        # logger.debug(f'row1=\n{row1[:5]}')
        row2 = np.hstack([zeros, model, -v * model])
        assert row2.shape == (model_h, 9)
        # logger.debug(f'row2=\n{row2[:5]}')
        L = np.vstack([row1, row2])
        logger.debug(f'用来求解单应性的系数矩阵 L=\n{L}')
        assert L.shape == (2 * model_h, 9)
        _, _, Vh = svd(L)
        m = Vh[-1, :]
        assert len(m) == 9  # 可以估计单应性的全部三个行向量
        estimated_homography = m.reshape((3,3))
        normalized_estimated_homography = estimated_homography / estimated_homography[2,2]
        return normalized_estimated_homography

    @classmethod
    def infer_homography_without_radial_distortion_with_isotropic_scaling(cls, model: np.ndarray, pixel: np.ndarray):
        # 数据准备：截取模型点的前两列 (X, Y)
        model_nonhomo = model[:, :2]
        pixel_nonhomo = pixel[:, :2]
        # 归一化 (Normalization)
        model_nonhomo_norm, T_model = normalize_points(model_nonhomo)
        pixel_nonhomo_norm, T_pixel = normalize_points(pixel_nonhomo)
        # 构建齐次坐标 (Homogeneous Coordinates)
        model_norm = np.hstack([model_nonhomo_norm, model[:, 2:3]])
        pixel_norm = np.hstack([pixel_nonhomo_norm, pixel[:, 2:3]])
        # 利用 svd 估计单应性
        homography_norm = cls.infer_homography_without_radial_distortion(model_norm, pixel_norm)
        # 去归一化 (Denormalization)
        # H = inv(T_pixel) * homography_norm * T_model
        # 这里的变换关系推导如下：
        # pixel_nonhomo_norm = homography_norm * model_nonhomo_norm
        # T_pixel * pixel = homography_norm * T_model * model
        # pixel = (inv(T_pixel) * homography_norm * T_model) * model
        homography = np.linalg.pinv(T_pixel) @ homography_norm @ T_model
        # 将 homography 的最后一个元素归一化为 1
        homography = homography / homography[2, 2]
        # 返回扁平化的向量，保持与接口一致 (9,)
        return homography

    @classmethod
    def extract_intrinsic_parameters_from_homography(cls, list_of_homography: typing.List[np.ndarray]):
        """
        把相机内部参数逐个提取出来
        """

        def v_constraint(homography: np.ndarray, i: int, j: int):
            return np.array([
                    # h_{i1} h_{j1}
                    homography[i, 0] * homography[j, 0],
                    # h_{i1} h_{j2} + h_{i2} h_{j1}
                    homography[i, 0] * homography[j, 1] + homography[i, 1] * homography[j, 0],
                    # h_{i2} h_{j2}
                    homography[i, 1] * homography[j, 1],
                    # h_{i3} h_{j1} + h_{i1} h_{j3}
                    homography[i, 2] * homography[j, 0] + homography[i, 0] * homography[j, 2],
                    # h_{i3} h_{j2} + h_{i2} h_{j3}
                    homography[i, 2] * homography[j, 1] + homography[i, 1] * homography[j, 2],
                    # h_{i3} h_{j3}
                    homography[i, 2] * homography[j, 2],
            ])

        V = np.vstack(list([
            v_constraint(homography, 0, 1),                                     # v_12.T
            v_constraint(homography, 0, 0) - v_constraint(homography, 1, 1),    # v_11.T - v_22.T
        ] for homography in list_of_homography))

        assert V.shape == (2 * len(list_of_homography), 6), f'矩阵 V 的形状实际上是: {V.shape}'

        _, _, Vh = svd(V.T @ V)
        b = Vh[-1, :]
        assert b.shape == (6,), f'向量 b 的形状实际上是: {b.shape}'
        B11, B12, B22, B13, B23, B33 = b
        BB = np.array([
            [B11, B12, B13],
            [B12, B22, B23],
            [B13, B23, B33],
        ])
        logger.debug(f'估计的基本矩阵 =\n{BB}')
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
        logger.debug(f'{v0=}')
        rho = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
        logger.debug(f'{rho=}')
        alpha0 = rho / B11
        logger.debug(f'{alpha0=}')
        alpha = np.sqrt(alpha0)
        logger.debug(f'{alpha=}')
        beta0 = rho * B11 / (B11 * B22 - B12 ** 2)
        logger.debug(f'{beta0=}')
        beta = np.sqrt(beta0)
        logger.debug(f'{beta=}')
        gamma = - B12 * alpha ** 2 * beta / rho
        logger.debug(f'{gamma=}')
        u0 = gamma * v0 / beta - B13 * alpha ** 2 / rho
        logger.debug(f'{u0=}')
        K = np.array([
            [alpha, gamma, u0],
            [.0, beta, v0],
            [.0, .0, 1.],
        ], dtype=np.float32)

        return K

    # @classmethod
    # def guess_homography_with_radial_distortion(cls, model: np.ndarray, pixel: np.ndarray):
    #     # 这里构造的矩阵 L 与张正友的报告中的不同，原因是这里考虑了径向畸变因素
    #     # L := [
    #     #     [v1 * M1, - u1 * M1],
    #     #     [v2 * M2, - u2 * M2],
    #     #     ...,
    #     #     [vn * Mn, - un * Mn],
    #     # ]
    #     model_h, model_w = model.shape
    #     u = pixel[:, 0]
    #     v = pixel[:, 1]
    #     vm = v[:, None] * model
    #     um = -u[:, None] * model
    #     L = np.hstack([vm, um])
    #     logger.debug(f'L=\n{L}')
    #     assert L.shape == (model_h, 2 * model_w)
    #     _, _, Vh = svd(L)
    #     m = Vh[-1, :]
    #     assert len(m) == 6, f'Invalid shape: {m.shape}'  # 可以估计的只有单应性的两个行向量
    #     return m.reshape((2,3))

    # @classmethod
    # def guess_homography_with_radial_distortion_with_isotropic_scaling(cls, model: np.ndarray, pixel: np.ndarray):
    #     """
    #     基于各向同性归一化 (Isotropic Scaling) 的
    #     “带径向畸变项的单应性估计” 的线性方程组构建（仅构建 L，不求解畸变系数）。

    #     返回值为长度 6 的向量，对应单应性矩阵的前两行。
    #     """

    #     # -------------------------
    #     # 1. 准备模型点 (只取 X,Y)
    #     # -------------------------
    #     model_2d = model[:, :2]       # (N,2)

    #     # -------------------------
    #     # 2. 分别对 model 和 pixel 做归一化
    #     # -------------------------
    #     model_norm, T_model = normalize_points(model_2d)   # (N,2)
    #     pixel_norm, T_pixel = normalize_points(pixel)      # (N,2)
    #     logger.debug(f'model_norm=\n{model_norm[:5]}')

    #     # 提取 u,v
    #     u = pixel_norm[:, 0:1]
    #     v = pixel_norm[:, 1:2]
    #     logger.debug(f'u=\n{u[:5]}')
    #     logger.debug(f'v=\n{v[:5]}')

    #     # -------------------------
    #     # 3. 构造归一化后的 L 矩阵
    #     #    L = [ v_i * M_i ,  -u_i * M_i ]
    #     # -------------------------
    #     # model_norm = [[X_i, Y_i], ...]
    #     vm = v * model_norm      # (N,2)
    #     um = -u * model_norm     # (N,2)

    #     # 拼成 L: (N,4)
    #     L = np.hstack([vm, um])

    #     # -------------------------
    #     # 4. 用 SVD 求最小奇异值对应的右奇异向量
    #     # -------------------------
    #     _, _, Vh = svd(L)
    #     m = Vh[-1, :]
    #     assert len(m) == 4

    #     # -------------------------
    #     # 5. 与 guess_homography_with_radial_distortion 一致：
    #     #    返回 (2,3) 形状的矩阵
    #     #    但 L 只有4列，因此补齐 0
    #     # -------------------------
    #     # 原始 m 预期长度 = 6，因此补 2 个 0
    #     m_full = np.hstack([
    #         m.reshape((2,2)),
    #         np.zeros((2,1))
    #     ])
    #     assert m_full.shape == (2,3)

    #     return m_full

def plot_relation_between_rotation_skewness_and_homography_relative_error_old(rotation_skewness_and_homography_relative_error):
    # 利用 matplotlib 绘制旋转偏斜度与单应性相对误差的散点图
    rotation_skewness, homography_relative_error = zip(*rotation_skewness_and_homography_relative_error)
    plt.figure(figsize=(10, 6))
    plt.scatter(rotation_skewness, homography_relative_error, color='blue', marker='o')
    plt.title('Rotation Skewness vs Homography Relative Error')
    plt.xlabel('Rotation Skewness (Frobenius Norm)')
    plt.ylabel('Homography Relative Error')
    plt.grid(True)
    plt.show()

def plot_relation_between_rotation_skewness_and_homography_relative_error(
    rotation_skewness_and_homography_relative_error,
    remove_outliers=False,
    eps=0.3,  # DBSCAN参数：邻域距离
    min_samples=10,  # DBSCAN参数：最小样本数
    y_max = None
):
    """
    绘制旋转偏斜度与单应性相对误差的散点图，可选是否去除异常点

    Args:
        rotation_skewness_and_homography_relative_error: 包含旋转偏斜度和单应性相对误差的二元组列表
        remove_outliers: 是否移除异常点，默认为False
        eps: DBSCAN算法的邻域距离参数
        min_samples: DBSCAN算法的最小样本数参数
    """
    # 解压数据
    rotation_skewness, homography_relative_error = zip(*rotation_skewness_and_homography_relative_error)

    # 转换为numpy数组便于处理
    rotation_skewness = np.array(rotation_skewness)
    homography_relative_error = np.array(homography_relative_error)

    # 初始化正常点索引（全部点）
    normal_point_indices = np.ones_like(homography_relative_error, dtype=bool)

    if y_max is not None:
        remove_outliers = False

    if remove_outliers:
        logger.info("开始检测异常点...")

        # 使用DBSCAN进行异常点检测
        # 重塑数据以适应DBSCAN输入要求
        X = homography_relative_error.reshape(-1, 1)
        print(f'{X.shape=}')

        # 使用DBSCAN聚类算法
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # 标记异常点（DBSCAN将噪声点标记为-1）
        outlier_mask = (labels == -1)
        normal_point_indices = ~outlier_mask

        # 计算正常点的y轴最大值
        if np.any(normal_point_indices):
            y_max = np.max(homography_relative_error[normal_point_indices])
            logger.info(f"主要部分单应性相对误差最大值: {y_max:.6f}")

        # 记录异常点信息
        outlier_count = np.sum(outlier_mask)
        if outlier_count > 0:
            logger.warning(f"检测到 {outlier_count} 个异常点:")
            for i in np.where(outlier_mask)[0]:
                logger.warning(
                    f"异常点 {i}: 旋转偏斜度 = {rotation_skewness[i]:.6f}, "
                    f"单应性相对误差 = {homography_relative_error[i]:.6f}"
                )
        else:
            logger.info("未检测到异常点")

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制散点图
    if remove_outliers and np.any(normal_point_indices):
        # 只绘制正常点
        plt.scatter(rotation_skewness[normal_point_indices],
                   homography_relative_error[normal_point_indices],
                   color='blue', marker='o', label='normal points')

    else:
        # 绘制所有点
        plt.scatter(rotation_skewness, homography_relative_error,
                   color='blue', marker='o')

    # 设置标题和标签
    plt.title('Rotation Skewness vs Homography Relative Error' +
              (' (without outliers)' if remove_outliers else ''))
    plt.xlabel('Rotation Skewness (Frobenius Norm)')
    plt.ylabel('Homography Relative Error')

    # 设置y轴范围
    plt.ylim(bottom=0)
    if y_max is not None:
        # 为y轴上限添加一些边距
        y_margin = y_max * 0.1  # 10%的边距
        plt.ylim(top=y_max + y_margin)

    # 添加网格和图例
    plt.grid(True, alpha=0.3)
    if remove_outliers and np.any(~normal_point_indices):
        plt.legend()

    plt.tight_layout()
    plt.show()




def run():
    projection_model = CameraModel(
        d=100,
        a=400,
        b=300,
        theta=90.5,
        u0=320,
        v0=240,
    )
    model_points = generate_model_points()
    n_photos = 100

    rotation_skewness_and_homography_relative_error = []
    list_of_homography = []

    for _ in range(n_photos):
        logger.debug('\n' * 5)

        _, image_points, rotation, _, homography = projection_model.randomly_project(model_points)
        # logger.debug(f'image_points=\n{image_points[:5]}')

        m1 = ZhangCameraCalibration.infer_homography_without_radial_distortion(model_points, image_points)
        # m1 = ZhangCameraCalibration.infer_homography_without_radial_distortion_with_isotropic_scaling(model_points, image_points)
        logger.debug(f'm1=\n{m1}')
        rotation_skewness_and_homography_relative_error.append((
            np.linalg.norm(rotation - np.eye(rotation.shape[0])),
            ZhangCameraCalibration.evaluate_relative_error(m1, homography),
        ))
        list_of_homography.append(m1)

    # plot_relation_between_rotation_skewness_and_homography_relative_error_old(
    #     rotation_skewness_and_homography_relative_error
    # )
    # plot_relation_between_rotation_skewness_and_homography_relative_error(
    #     rotation_skewness_and_homography_relative_error,
    #     remove_outliers=True,
    # )
    # plot_relation_between_rotation_skewness_and_homography_relative_error(
    #     rotation_skewness_and_homography_relative_error,
    #     y_max=.004,
    # )

    logger.debug('\n' * 5)
    K = ZhangCameraCalibration.extract_intrinsic_parameters_from_homography(list_of_homography)
    logger.debug(f'估计的相机内参矩阵 K=\n{K}')
    realK = projection_model.K
    realKinv = np.linalg.inv(realK)
    logger.debug(f'真实的基本矩阵 =\n{realKinv.T @ realKinv}')




if __name__ == '__main__':
    run()
