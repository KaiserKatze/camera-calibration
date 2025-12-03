#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import functools
import typing
import time

import numpy as np
import scipy.optimize
import scipy
import scipy.io
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

    formatter = logging.Formatter('%(message)s')  # 创建一个formatter，用于定义日志格式
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


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        logger.debug(f'计算耗时: {self.elapsed_time:.6f} 秒')



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
    points_2d = np.array(points_2d, dtype=np.float64)

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


def print_all_conditions_of_matrix(matrix: np.ndarray, name: str) -> None:
    """
    依次打印矩阵 matrix 的列和条件数、谱条件数、行和条件数和 F 条件数。

    Args
        matrix 需要打印条件数的矩阵
        name 矩阵变量名
    """
    cond_1 = np.linalg.cond(matrix, 1)  # L1 范数
    cond_2 = np.linalg.cond(matrix, 2)  # L2 范数
    cond_inf = np.linalg.cond(matrix, np.inf) # L_infty 范数
    cond_fro = np.linalg.cond(matrix, 'fro')  # F 范数
    cond_max = np.max([ cond_1, cond_2, cond_inf, cond_fro ])
    ill_max = '病态' if cond_max > 1 else ''
    logger.debug(f'''矩阵 {name} 的条件数：
    列和条件数 \t= {cond_1:.8f}
    谱条件数   \t= {cond_2:.8f}
    行和条件数 \t= {cond_inf:.8f}
    F 条件数   \t= {cond_fro:.8f}
    最大值    \t= {cond_max:.8f}\t{ill_max}''')


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
        Tz = np.random.uniform(1000, 1500)
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
        # logger.debug('\n' * 5)
        # logger.debug(f'模型点的二维齐次坐标 model_2d_homo=\n{model_2d_homo[:5]}')
        # 计算单应性 H
        H = self.K @ np.hstack((rotation[:, :2], translation))
        assert H.shape == (3, 3)
        # logger.debug(f'真实的旋转矩阵 R=\n{rotation}')
        # logger.debug(f'真实的平移向量 T=\n{translation}')
        # logger.debug(f'真实的单应性 H=\n{H}')
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
        # logger.debug(f'用来求解单应性的系数矩阵 L=\n{L}')
        assert L.shape == (2 * model_h, 9)

        _, _, Vh = svd(L)
        m = Vh[-1, :]
        assert len(m) == 9  # 可以估计单应性的全部三个行向量
        estimated_homography = m.reshape((3,3))
        normalized_estimated_homography = estimated_homography / estimated_homography[2,2]
        return normalized_estimated_homography

    @classmethod
    def infer_homography_without_radial_distortion_with_zscore_scaling(cls, model: np.ndarray, pixel: np.ndarray):
        # 归一化 (Normalization)
        model_nonhomo = model[:, :2]
        pixel_nonhomo = pixel[:, :2]
        mean_model = np.mean(model_nonhomo, axis=0)
        std_model = np.std(model_nonhomo, axis=0)
        mean_pixel = np.mean(pixel_nonhomo, axis=0)
        std_pixel = np.std(pixel_nonhomo, axis=0)
        model_nonhomo_norm = (model_nonhomo - mean_model) / std_model
        pixel_nonhomo_norm = (pixel_nonhomo - mean_pixel) / std_pixel
        # 构建齐次坐标 (Homogeneous Coordinates)
        model_norm = np.hstack([model_nonhomo_norm, model[:, 2:3]])
        pixel_norm = np.hstack([pixel_nonhomo_norm, pixel[:, 2:3]])
        # 利用 svd 估计单应性
        homography = cls.infer_homography_without_radial_distortion(model_norm, pixel_norm)
        # 将 homography 的最后一个元素归一化为 1
        homography = homography / homography[2, 2]
        return homography

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
        return homography

    @staticmethod
    def project_to_positive_definite_matrix(A: np.ndarray, min_eig: float = 1e-8):
        """把对称矩阵 A 投影为最近的对称正定矩阵（通过特征值裁剪）。"""
        assert A.shape[0] == A.shape[1]
        eigen_values, eigen_vectors = np.linalg.eigh(A)
        eigen_values_clipped = np.clip(eigen_values, a_min=min_eig, a_max=None)
        A_pd = (eigen_vectors * eigen_values_clipped) @ eigen_vectors.T
        return 0.5 * (A_pd + A_pd.T)

    @classmethod
    def extract_intrinsic_parameters_from_homography(cls, list_of_homography: typing.List[np.ndarray],
                                                     model_2d_homo: np.ndarray,
                                                     list_of_pixel_2d_homo: typing.List[np.ndarray]):
        """
        把相机内部参数逐个提取出来
        """

        def v_constraint(homography: np.ndarray, i: int, j: int):
            def h(x: int, y: int):
                return homography[x-1, y-1]

            return np.array([
                h(i, 1) * h(j, 1),
                h(i, 1) * h(j, 2) + h(i, 2) * h(j, 1),
                h(i, 2) * h(j, 2),
                h(i, 3) * h(j, 1) + h(i, 1) * h(j, 3),
                h(i, 3) * h(j, 2) + h(i, 2) * h(j, 3),
                h(i, 3) * h(j, 3),
            ])

        V = np.vstack(list([
            v_constraint(homography, 1, 2),                                     # v_12.T
            v_constraint(homography, 1, 1) - v_constraint(homography, 2, 2),    # v_11.T - v_22.T
        ] for homography in list_of_homography))

        assert V.shape == (2 * len(list_of_homography), 6), f'矩阵 V 的形状实际上是: {V.shape}'

        # 计算矩阵 V 的条件数
        # print_all_conditions_of_matrix(V.T @ V, '(V.T @ V)')

        _, S, Vh = svd(V)
        abs_singular_value = abs(S)
        min_abs_singular_value = abs_singular_value.min()
        max_abs_singular_value = abs_singular_value.max()
        logger.debug(f'矩阵 V 的奇异值 = \n\t{ ','.join('{:.6f}'.format(x) for x in S.tolist()) }\n'
                     + f'\t奇异值绝对值最大值 = \t{ max_abs_singular_value :.6f}\n'
                     + f'\t奇异值绝对值最小值 = \t{ min_abs_singular_value :.6f}\n'
                     + f'\t矩阵 V 的谱条件数 = \t{ max_abs_singular_value / min_abs_singular_value :2f}')

        b = Vh[-1, :]
        assert b.shape == (6,), f'向量 b 的形状实际上是: {b.shape}'
        B11, B12, B22, B13, B23, B33 = b / np.sign(b[0])
        BB = np.array([
            [B11, B12, B13],
            [B12, B22, B23],
            [B13, B23, B33],
        ])
        logger.debug(f'估计的基本矩阵 =\n{BB}')
        principal_minors = [np.linalg.det(BB[:i, :i]) for i in range(1,4)]
        principal_minors_str = ', '.join(['{:.6f}'.format(x) for x in principal_minors])
        logger.debug(f'估计的基本矩阵的顺序主子式 =\n{principal_minors_str}')
        if not all(pm > 0 for pm in principal_minors):
            logger.warning('估计的基本矩阵不是正定矩阵，可能存在数值不稳定问题！')
            BB = cls.project_to_positive_definite_matrix(BB)
            principal_minors = [np.linalg.det(BB[:i, :i]) for i in range(1,4)]
            principal_minors_str = ', '.join(['{:.6f}'.format(x) for x in principal_minors])
            logger.debug(f'重映射得到的基本矩阵的顺序主子式 =\n{principal_minors_str}')
            B11 = BB[0, 0]
            B12 = BB[0, 1]
            B13 = BB[0, 2]
            B22 = BB[1, 1]
            B23 = BB[1, 2]
            B33 = BB[2, 2]

        mid1 = B12 * B13 - B11 * B23
        mid2 = B11 * B22 - B12 ** 2
        logger.debug(f'{mid1=:.6f}')
        logger.debug(f'{mid2=:.6f}')
        v0 = mid1 / mid2
        logger.debug(f'{v0=:.6f}')
        rho = B33 - (B13 ** 2 + v0 * mid1) / B11
        alpha0 = rho / B11
        if alpha0 < 0:
            alpha0 = -alpha0
            rho = -rho
            logger.debug('估计的 alpha0 为负数，已取其绝对值进行计算！')
        logger.debug(f'{rho=:.6f}')
        logger.debug(f'{alpha0=:.6f}')
        alpha = np.sqrt(alpha0)
        logger.debug(f'{alpha=:.6f}')
        beta0 = rho * B11 / mid2
        logger.debug(f'{beta0=:.6f}')
        beta = np.sqrt(beta0)
        logger.debug(f'{beta=:.6f}')
        gamma = - B12 * alpha0 * beta / rho
        logger.debug(f'{gamma=:.6f}')
        u0 = gamma * v0 / beta - B13 * alpha0 / rho
        logger.debug(f'{u0=:.6f}')
        K = np.array([
            [alpha, gamma, u0],
            [.0, beta, v0],
            [.0, .0, 1.],
        ], dtype=np.float64)
        logger.debug(f'估计的相机内参矩阵 (initial guess) K=\n{K}')

        # 用牛顿法
        def residuals(x) -> np.float64:
            Kinv = np.linalg.pinv(x.reshape((3,3)))
            pixel_2d_homo_diff = []
            for homography, pixel_2d_homo in zip(list_of_homography, list_of_pixel_2d_homo):
                h1, h2, h3 = homography.T
                KinvH1 = Kinv @ h1
                KinvH2 = Kinv @ h2
                KinvH3 = Kinv @ h3
                denom = 1.0 / np.linalg.norm(KinvH1)
                r1 = KinvH1 * denom
                r2 = KinvH2 * denom
                r3 = np.cross(r1, r2)
                t = denom * KinvH3

                # 正交化
                R = np.column_stack([r1, r2, r3])
                U, _, Vt = svd(R)
                R = U @ Vt
                r1, r2, r3 = R.T  # 提取列向量

                pixel_2d_homo_pred = model_2d_homo @ np.column_stack([r1, r2, t]).T
                pixel_2d_homo_diff.append(pixel_2d_homo - pixel_2d_homo_pred)

            pixel_2d_homo_diff = np.array(pixel_2d_homo_diff)
            pixel_2d_homo_diff = np.linalg.norm(pixel_2d_homo_diff, axis=1)
            return pixel_2d_homo_diff.reshape(-1)

        optimize_result = scipy.optimize.least_squares(
            residuals,
            x0=K.reshape(-1),
            method='lm',  # Levenberg-Marquardt algorithm
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=5000,
            verbose=2,
        )

        print(f'{optimize_result=}')
        x_opt = optimize_result.x
        assert x_opt.shape == (9,)
        K = x_opt.reshape((3,3))

        return K



def save_mat(path: str, model_2d_homo: np.ndarray,
             image_size: typing.Tuple[int, int],
             list_of_pixel_2d_homo: typing.List[np.ndarray],
             real_intrinsic_matrix: np.ndarray,
             list_of_rotation: typing.List[np.ndarray],
             list_of_translation: typing.List[np.ndarray]) -> None:
    scipy.io.savemat(
        path,
        {
            'model_2d_homo': model_2d_homo,
            'image_size': image_size,
            'list_of_pixel_2d_homo': list_of_pixel_2d_homo,
            'real_intrinsic_matrix': real_intrinsic_matrix,
            'list_of_rotation': list_of_rotation,
            'list_of_translation': list_of_translation,
        },
    )


def load_mat(path: str) -> typing.Dict[str, np.ndarray]:
    return scipy.io.loadmat(path)


def infer_image_size(list_of_image_points, margin=2, min_size=(480, 640)):
    """
    输入:
      list_of_image_points: 可以是
         - numpy.ndarray with shape (V, M, 3)  (your zhang.mat: 1000x256x3)
         - list of arrays each shape (M, 3)
         - single array shape (M,3)
      margin: 整数，结果上再加的像素余量
      min_size: (min_height, min_width) 最小尺寸下限
    返回:
      (height, width) 两个整数
    """
    pts = list_of_image_points

    # 转为 numpy 数组（若是 list，尝试 stack）
    if isinstance(pts, list):
        pts = np.stack([np.asarray(p) for p in pts], axis=0)   # -> (V, M, 3)
    else:
        pts = np.asarray(pts)

    # 可能的形状：
    # (V, M, 3)  <-- 1000 x 256 x 3
    # (M, 3, V)  etc. 处理常见变体
    if pts.ndim == 3 and pts.shape[2] == 3:
        arr = pts  # (V, M, 3)
    elif pts.ndim == 3 and pts.shape[1] == 3:
        # 例如 (V, 3, M) or (something,3,M) -> 转置为 (V, M, 3)
        arr = pts.transpose(0, 2, 1)
    elif pts.ndim == 2 and pts.shape[1] == 3:
        # 单幅图像 (M,3) -> 加一个视图维度
        arr = pts[np.newaxis, ...]
    else:
        raise ValueError(f'unrecognized shape for list_of_image_points: {pts.shape}')

    # 把齐次坐标除以第三个分量得到非齐次 u,v
    # 防止除以0 (理论上第三分量都是1)，用 eps 保护
    denom = arr[..., 2:3]
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    uv = arr[..., :2] / denom   # shape (V, M, 2)

    # 取所有视图与所有点的最大 u (x) 和 v (y)
    max_u = np.nanmax(uv[..., 0])
    max_v = np.nanmax(uv[..., 1])

    width  = int(np.ceil(max_u)) + int(margin)
    height = int(np.ceil(max_v)) + int(margin)

    # 应用最小尺寸下限
    height = max(height, int(min_size[0]))
    width  = max(width,  int(min_size[1]))

    return (height, width)


def init():
    projection_model = CameraModel(
        d=100,
        a=400,
        b=300,
        theta=np.radians(90.5),  # 将角度制的 90.5° 转为弧度制
        u0=320,
        v0=240,
    )
    model_points = generate_model_points()
    n_photos = 100

    list_of_image_points = []
    list_of_rotation = []
    list_of_translation = []

    for _ in range(n_photos):
        _, image_points, rotation, translation, _ = projection_model.randomly_project(model_points)
        list_of_image_points.append(image_points)
        list_of_rotation.append(rotation)
        list_of_translation.append(translation)

    image_size = infer_image_size(list_of_image_points)

    save_mat(
        'zhang.mat',
        model_points,           # 真实的模型点二维齐次坐标
        image_size,
        list_of_image_points,
        projection_model.K,     # 真实的相机内参矩阵
        list_of_rotation,       # 真实的旋转矩阵
        list_of_translation,    # 真实的平移向量
    )


def compare_with_opencv():
    logger.debug('\n' * 10 + '=' * 100)
    import cv2
    saved_data = load_mat('zhang.mat')
    model_points_h = saved_data['model_2d_homo']             # (M,3) 齐次模型点
    raw_list_of_image_points = saved_data['list_of_pixel_2d_homo']  # 可能是 list/ndarray

    # 准备 objectPoints: 将模型点变为 (M,3) 非齐次（平面 z=0）
    model_nonhomo = model_points_h[:, :2].astype(np.float32)
    # 把平面模型作为 3D 点，z=0
    objp3 = np.hstack([model_nonhomo, np.zeros((model_nonhomo.shape[0], 1), dtype=np.float32)])  # (M,3)

    # 规范化 list_of_image_points 的容器形式 -> 得到一个 python list，每项为 (M,3) 或 (M,2)
    image_points_list = []
    # loadmat 有时把 list 存为 ndarray of object; 统一转换
    if isinstance(raw_list_of_image_points, np.ndarray) and raw_list_of_image_points.dtype == np.object_:
        # 这是 scipy.io.loadmat 存储 list 的常见形式
        # 遍历每个单元并转换为 numpy 数组
        for item in raw_list_of_image_points.ravel():
            pts = np.asarray(item)
            # 有时会是 (M,3) 或 (1,M,3) 等，先压平到 (M,3)
            if pts.ndim == 3 and pts.shape[0] == 1:
                pts = pts[0]
            image_points_list.append(pts)
    elif isinstance(raw_list_of_image_points, list):
        image_points_list = [np.asarray(p) for p in raw_list_of_image_points]
    else:
        # 可能是 ndarray (V, M, 3)
        image_points_list = [np.asarray(raw_list_of_image_points[i]) for i in range(raw_list_of_image_points.shape[0])]

    # 将每个 image points 转成 (M,2) 非齐次坐标 (u,v)
    imagePoints_for_cv = []
    for pts in image_points_list:
        pts = np.asarray(pts)
        # 常见情形: (M,3) 齐次坐标
        if pts.ndim == 2 and pts.shape[1] == 3:
            denom = pts[:, 2:3]
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            uv = pts[:, :2] / denom
        # 也可能是 (M,2)
        elif pts.ndim == 2 and pts.shape[1] == 2:
            uv = pts
        # 也可能是 (1, M, 3) 或 (M, 3, 1) 等，尝试 reshape
        else:
            pts_flat = pts.reshape(-1, pts.shape[-1])
            if pts_flat.shape[1] == 3:
                denom = pts_flat[:, 2:3]
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
                uv = pts_flat[:, :2] / denom
            elif pts_flat.shape[1] == 2:
                uv = pts_flat
            else:
                raise ValueError(f'unrecognized image points shape: {pts.shape}')
        imagePoints_for_cv.append(uv.astype(np.float32))

    # objectPoints: 对每一张图都复制同样的模型点（因为是标定板）
    objectPoints_for_cv = [objp3.copy() for _ in imagePoints_for_cv]

    # 取 image_size （loadmat 载入后可能为 array）
    image_size = saved_data.get('image_size')
    if isinstance(image_size, np.ndarray):
        # 可能是 (2,1) 或 (1,2) 等
        image_size = tuple(map(int, np.asarray(image_size).ravel()))
    image_height, image_width = image_size  # 在脚本中 infer_image_size 返回 (height, width)
    image_size_cv = (int(image_width), int(image_height))  # OpenCV expects (width, height)

    # 调用 OpenCV calibrateCamera
    # 可选 flags，根据需要你可以固定某些畸变项；这里不做特别固定，用默认让算法估计
    flags = 0
    # 如果你只想估计内参而不估计高阶径向畸变，可以使用：
    # flags = cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

    # 使用初始内参为 None，让 calibrateCamera 自行初始化
    with Timer():
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints_for_cv,
            imagePoints_for_cv,
            image_size_cv,
            None,
            None,
            flags=flags
        )

    logger.debug(f'OpenCV calibrateCamera 返回值 ret={ret}')
    logger.debug(f'估计的相机内参矩阵 K=\n{camera_matrix}')
    logger.debug(f'估计的畸变系数 dist_coeffs =\n{dist_coeffs.ravel()}')

    # 计算每张图的重投影误差并求总均方根误差 (RMS)
    total_error = 0.0
    total_points = 0
    for i in range(len(objectPoints_for_cv)):
        objp = objectPoints_for_cv[i]
        imgp = imagePoints_for_cv[i]
        rvec = rvecs[i]
        tvec = tvecs[i]
        # projectPoints 输出 (N,1,2)
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        err = imgp - projected
        total_error += np.sum(np.linalg.norm(err, axis=1) ** 2)
        total_points += objp.shape[0]

    rmse = np.sqrt(total_error / total_points)
    logger.info(f'OpenCV 总重投影 RMSE = {rmse:.6f} 像素')

    # 与真实内参比较（如果存在）
    realK = saved_data.get('real_intrinsic_matrix')
    if realK is not None:
        try:
            realK = np.asarray(realK, dtype=np.float32)
            # 如果 loadmat 导致维度多一级，扁平化
            if realK.shape != (3,3):
                realK = realK.reshape(3,3)
            # logger.info(f'真实的相机内参 K =\n{realK}')
            # logger.info(f'估计 K 与 真实 K 差异 =\n{camera_matrix - realK}')
            # 相对误差
            rel_err = np.linalg.norm(camera_matrix - realK) / (np.linalg.norm(realK) + 1e-12)
            logger.info(f'估计 K 相对误差 = {rel_err*100:.6f}%')
        except Exception:
            logger.warning('无法解析 real_intrinsic_matrix 的形状以用于比较。')

    # 返回一些有用结果，便于后续程序使用或测试
    return {
        'ret': ret,
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'rmse': rmse,
    }



def run(infer_homography_fn: typing.Callable):
    saved_data = load_mat('zhang.mat')
    model_points = saved_data['model_2d_homo']
    list_of_image_points = saved_data['list_of_pixel_2d_homo']
    with Timer():
        list_of_homography = []
        for image_points in list_of_image_points:
            list_of_homography.append(
                infer_homography_fn(
                    model_points, image_points
                )
            )
        K = ZhangCameraCalibration.extract_intrinsic_parameters_from_homography(
            list_of_homography, model_points, list_of_image_points
        )
    logger.debug(f'估计的相机内参矩阵 K=\n{K}')
    realK = saved_data['real_intrinsic_matrix']
    relative_error_of_intrinsic_matrix = np.linalg.norm(K - realK) / np.linalg.norm(realK)
    logger.debug(f'相机内参矩阵相对误差 =\n{relative_error_of_intrinsic_matrix*100:.6f}%')




if __name__ == '__main__':
    # init()  # 生成模型点和像素点

    # 用我自己实现的算法，求解相机内参矩阵
    saved_data = load_mat('zhang.mat')
    realK = saved_data['real_intrinsic_matrix']
    realKinv = np.linalg.inv(realK)
    logger.debug(f'真实的相机内参矩阵 K=\n{realK}')
    logger.debug(f'真实的基本矩阵 =\n{realKinv.T @ realKinv}')

    for fn_index, fn in enumerate([
        # ZhangCameraCalibration.infer_homography_without_radial_distortion,
        ZhangCameraCalibration.infer_homography_without_radial_distortion_with_isotropic_scaling,
        # ZhangCameraCalibration.infer_homography_without_radial_distortion_with_zscore_scaling,
    ]):
        logger.debug('\n' * 10 + '=' * 100)
        logger.debug(f'尝试第 {fn_index} 种方法，推断单应性 ...')
        run(fn)

    # 使用 opencv 现有的算法，求解相机内参矩阵
    # compare_with_opencv()
