#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import functools
import typing
import time
import os

import numpy as np
import scipy.optimize
import scipy
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



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
        logger.debug(f'计算耗时: {self.elapsed_time:.3f} 秒')



def skew(v: np.ndarray) -> np.ndarray:
    """返回向量 v=(x,y,z) 的反对称矩阵 [v]_x"""
    x, y, z = v.ravel()
    return np.array([[0.0, -z,  y],
                     [ z, 0.0, -x],
                     [-y,  x, 0.0]], dtype=np.float64)

def rodrigues(r: np.ndarray) -> np.ndarray:
    """
    Rodrigues: 3-vector -> 3x3 rotation matrix
    r: shape (3,) or (3,1)
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if r.size != 3:
        raise ValueError(f'rodrigues: expected length-3 vector, got length {r.size}')
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        # 小角近似：R ≈ I + [r]_x
        return np.eye(3, dtype=np.float64) + skew(r)
    k = r / theta
    K = skew(k)
    c = np.cos(theta)
    s = np.sin(theta)
    R = c * np.eye(3) + (1 - c) * np.outer(k, k) + s * K
    return R

def inv_rodrigues(R: np.ndarray) -> np.ndarray:
    """
    逆 Rodrigues: 3x3 rotation matrix -> 3-vector
    返回 r: shape (3,)
    对 theta≈0 和 theta≈pi 做了数值保护
    """
    R = np.asarray(R, dtype=np.float64)
    assert R.shape == (3,3)
    # 计算角度
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    # 数值裁剪
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-12:
        # theta ~ 0, r ≈ vee(R - I)/2
        r = 0.5 * np.array([R[2,1] - R[1,2],
                             R[0,2] - R[2,0],
                             R[1,0] - R[0,1]], dtype=np.float64)
        return r

    # 若 theta ~ pi，需要特殊处理
    if abs(np.pi - theta) < 1e-6:
        # 使用对角元来构造旋转轴
        # 找到最大的对角元
        R_plus = (R + np.eye(3)) / 2.0
        # 取对角最大元素对应的轴分量
        axis = np.array([np.sqrt(max(R_plus[0,0], 0.0)),
                         np.sqrt(max(R_plus[1,1], 0.0)),
                         np.sqrt(max(R_plus[2,2], 0.0))], dtype=np.float64)
        # 正确符号根据非对角元素确定
        if axis[0] > 1e-8:
            axis[0] = np.sign(R[2,1] - R[1,2]) * axis[0]
        if axis[1] > 1e-8:
            axis[1] = np.sign(R[0,2] - R[2,0]) * axis[1]
        if axis[2] > 1e-8:
            axis[2] = np.sign(R[1,0] - R[0,1]) * axis[2]
        # 返回 r = theta * axis
        return theta * axis

    # 常规情形
    r_vec = (theta / (2.0 * np.sin(theta))) * np.array([R[2,1] - R[1,2],
                                                       R[0,2] - R[2,0],
                                                       R[1,0] - R[0,1]], dtype=np.float64)
    return r_vec



def normalize_points(points):
    """
    实现论文 [In defence of the 8-point algorithm Section 6.1](https://ieeexplore.ieee.org/document/466816) 描述的各向同性归一化 (Isotropic Scaling)。

    Args:
        points: 形状为 (N, 2) 的图像点坐标数组。

    Returns:
        new_points: 归一化后的点坐标，形状为 (N, 2)。
        T: 3x3 的变换矩阵，满足 new_points_homogeneous = T * old_points_homogeneous。
    """
    # 1. 计算重心 (Centroid)
    centroid = np.mean(points, axis=0)
    cx, cy = centroid[0], centroid[1]

    # 将点平移到原点
    shifted_points = points - centroid

    # 2. 计算平均距离
    # 计算每个点到原点的欧几里得距离
    distances = np.sqrt(np.sum(shifted_points**2, axis=1))
    mean_dist = np.mean(distances)

    # 3. 计算缩放因子，使得平均距离等于 sqrt(2)
    if mean_dist < 1e-8:
        scale = 1.0  # 防止除以接近0的值
    else:
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
    一共生成 N = ni * nj 个点。

    Returns:
        points_2d_homo: 模型点齐次坐标，形状为 (N, 3)。
    """

    ni = 10  # 网格列数
    nj = 14  # 网格行数
    a = 10  # 每个黑色正方形的边长（像素）

    points_2d = []
    for i in range(ni):
        for j in range(nj):
            # 计算当前正方形的左上角坐标
            points_2d.append(
                (i * a, j * a)
            )
    assert len(points_2d) == ni * nj
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
    assert points_2d_homo.shape[1] == 3
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
    列和条件数 \t= {cond_1:.6e}
    谱条件数   \t= {cond_2:.6e}
    行和条件数 \t= {cond_inf:.6e}
    F 条件数   \t= {cond_fro:.6e}
    最大值    \t= {cond_max:.6e}\t{ill_max}''')


def homo2nonhomo(points: np.ndarray) -> np.ndarray:
    """
    将齐次坐标 (X,Y,W) 转换为非齐次坐标 (u,v)

    :param points: 点的齐次坐标
    :type points: np.ndarray
    :return: 点的非齐次坐标
    :rtype: NDArray[float64]
    """
    assert points.ndim == 2 and points.shape[-1] == 3, f'{points.ndim=}, {points.shape=}'
    denom = points[:, 2:3]  # 齐次分量
    # 防止除以接近0的值，使用一个很小的数
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    points = points[:, :2] / denom
    return points


def assert_condition_number(list_of_homography: list[np.ndarray], cond_threshold: float = 1e6) -> list[int]:
    """
    根据单应性 H 的条件数筛选视图

    :param list_of_homography: 单应性列表
    :type list_of_homography: list[np.ndarray]
    :param cond_threshold: 条件数
    :type cond_threshold: float
    :return: 需要保留的单应性在列表中的序号
    :rtype: list[int]
    """

    kept_idx = []
    for idx, homography in enumerate(list_of_homography):
        # 计算条件数
        try:
            cond_H = np.linalg.cond(homography)
        except Exception:
            cond_H = np.inf

        if cond_H < cond_threshold:
            kept_idx.append(idx)

    remain_ratio = len(kept_idx) / max(1, len(list_of_homography))
    logger.debug(f'条件数筛选剩余率 = {remain_ratio * 100:.2f}%')
    return kept_idx


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
        Rx = np.random.uniform(-5, 5)
        Ry = np.random.uniform(-5, 5)
        Rz = np.random.uniform(-5, 5)
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
        Tx = np.random.uniform(-5, 5)
        Ty = np.random.uniform(-5, 5)
        Tz = np.random.uniform(10, 20)
        return cls(Tx, Ty, Tz)


class CameraModel:
    def __init__(self, d: float, a: float, b: float, theta: float, u0: float, v0: float):
        alpha = d / a
        beta = d / b
        logger.warning('当前实现没有使用 `theta` 参数!')
        self.K = np.array([
            # [alpha, -alpha / np.tan(theta), u0],
            [alpha, 0, u0],
            # [0, beta / np.sin(theta), v0],
            [0, beta, v0],
            [0, 0, 1],
        ], dtype=np.float32)

    @staticmethod
    def make_homography(intrinsic_matrix: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        依据相机内参矩阵、旋转矩阵、平移向量，计算“单应性”。
        这里产出的“单应性”只能与 Z=0 的模型点坐标相乘！

        :param intrinsic_matrix: 相机内参矩阵
        :type intrinsic_matrix: np.ndarray
        :param rotation: 旋转矩阵
        :type rotation: np.ndarray
        :param translation: 平移向量
        :type translation: np.ndarray
        :return: 单应性
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        return intrinsic_matrix @ np.hstack((rotation[:, :2], translation))

    def _arbitrary_project(self, model_2d_homo: np.ndarray,
                           rotation: np.ndarray,
                           translation: np.ndarray,
                           noise: float | None = 0.5):
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
        H = self.make_homography(self.K, rotation, translation)
        assert H.shape == (3, 3)
        logger.debug(f'真实的旋转矩阵 R=\n{rotation}\n\tdet(R)={np.round(np.linalg.det(rotation), 4)}')
        logger.debug(f'真实的平移向量 T=\n{translation}')
        logger.debug(f'真实的单应性 H=\n{H}')

        # 检查H矩阵是否接近奇异（条件数过大）
        cond_H = np.linalg.cond(H)
        logger.debug(f'单应性 H 条件数=\n{cond_H}')
        if cond_H > 1e12:
            logger.warning(f'单应性矩阵条件数过大: {cond_H:.6e}, 可能导致数值不稳定')

        # 计算像素点的齐次坐标
        pixel_2d_homo = model_2d_homo @ H.T

        # 检查是否有无穷大或NaN值
        if np.any(np.isinf(pixel_2d_homo)) or np.any(np.isnan(pixel_2d_homo)):
            raise ValueError('投影结果包含无穷大或NaN值')

        if noise is not None:
            # 计算像素点的非齐次坐标
            pixel_w = pixel_2d_homo[:, 2:3]
            pixel_2d_nonhomo = pixel_2d_homo[:, :2] / pixel_w
            # 给像素点的非齐次坐标添加高斯噪声（均值=0，标准差=0.5 像素）
            logger.debug('正在添加高斯噪声 ...')
            noise = np.random.normal(0, 0.5, pixel_2d_nonhomo.shape)
            pixel_2d_nonhomo += noise
            # 重新组装像素点的齐次坐标
            pixel_2d_homo = np.hstack((pixel_2d_nonhomo * pixel_w, pixel_w))

        assert pixel_2d_homo.shape == model_2d_homo.shape
        return model_2d_homo, pixel_2d_homo, rotation, translation, H

    def randomly_project(self, model_2d_homo: np.ndarray, noise=None):
        # 随机生成旋转矩阵和位移向量
        rotation = Rotation.randomize().R
        translation = Translation.randomize().T
        return self._arbitrary_project(model_2d_homo, rotation, translation, noise)

    def identity_project(self, model_2d_homo: np.ndarray):
        rotation = np.identity(3, dtype=np.float64)
        translation = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
        return self._arbitrary_project(model_2d_homo, rotation, translation, noise=None)

    @staticmethod
    def visualize_projection(objpoints: np.ndarray, imgpoints: list[np.ndarray],
                             view_index: int = 0, path_fig: str = None):
        """
        可视化第 view_index 个视图的模型点与投影点
        """
        # print('objpoints=\n', objpoints)
        # print('imgpoints=\n', imgpoints[view_index])
        objpoints = objpoints[:, 0:2]
        imgpoints = imgpoints[view_index, :, 0:2]
        obj_obs = objpoints.reshape(-1, 2)
        img_obs = imgpoints.reshape(-1, 2)
        assert img_obs.shape[0] == obj_obs.shape[0]

        plt.figure()
        plt.scatter(img_obs[:, 0], img_obs[:, 1], c='r', marker='o', label='像素点')
        plt.scatter(obj_obs[:, 0], obj_obs[:, 1], c='b', marker='x', label='模型点')
        for i in range(img_obs.shape[0]):
            plt.plot(
                [img_obs[i, 0], obj_obs[i, 0]],
                [img_obs[i, 1], obj_obs[i, 1]],
                'g-', linewidth=0.5,
            )
        plt.gca().invert_yaxis()  # 图像坐标通常原点在左上
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.title(f"第 {view_index} 个机位下 模型点-像素点 映射关系")
        if path_fig is None:
            path_fig = f'fig-{view_index}-projection.png'
        logger.debug(f'saving figure to: {os.path.abspath(path_fig)!r}')
        plt.savefig(path_fig, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def visualize_reprojection(objpoints: np.ndarray, imgpoints: list[np.ndarray],
                               estimated_intrinsic_matrix: np.ndarray,
                               rvecs: list[np.ndarray], tvecs: list[np.ndarray],
                               view_index: int = 0, path_fig: str = None):
        K = estimated_intrinsic_matrix
        rv = rvecs[view_index]
        tv = tvecs[view_index]
        tv = tv.reshape(3, 1)
        R = rodrigues(rv)  # 利用旋转参数 rv 构造旋转矩阵 R。
        H = CameraModel.make_homography(K, R, tv)  # 利用相机内参矩阵 K、旋转矩阵 R 和平移向量 tv 构造单应性 H。
        reprojection_points_homo = objpoints @ H.T  # 重投影，产出像素点的齐次坐标
        # print('rpipoints=\n', reprojection_points_homo)
        # print('imgpoints=\n', imgpoints[view_index])
        rpipoints = reprojection_points_homo[:, 0:2] / reprojection_points_homo[:, 2:3]
        imgpoints = imgpoints[view_index][:, 0:2] / imgpoints[view_index][:, 2:3]
        assert rpipoints.shape[0] == imgpoints.shape[0]

        plt.figure()
        plt.scatter(imgpoints[:, 0], imgpoints[:, 1], c='b', marker='x', label='观测的像素点')
        plt.scatter(rpipoints[:, 0], rpipoints[:, 1], c='r', marker='o', label='重投影像素点')
        for i in range(imgpoints.shape[0]):
            plt.plot(
                [imgpoints[i, 0], rpipoints[i, 0]],
                [imgpoints[i, 1], rpipoints[i, 1]],
                'g-', linewidth=0.5,
            )
        plt.gca().invert_yaxis()  # 图像坐标通常原点在左上
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.title(f"第 {view_index} 个机位下 重投影 映射关系")
        if path_fig is None:
            path_fig = f'fig-{view_index}-reprojection.png'
        logger.debug(f'saving figure to: {os.path.abspath(path_fig)!r}')
        plt.savefig(path_fig, dpi=150, bbox_inches='tight')
        plt.close()


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
        if not (model.shape[1] == pixel.shape[1] == 3):
            raise ValueError('必须输入模型点、像素点的二维齐次坐标!')
        if not (model.shape[0] == pixel.shape[0] > 6):
            logger.warning('样本数量太少!')
        model_h = model.shape[0]
        # 检查输入是否包含无穷大或NaN值
        if np.any(np.isinf(model)) or np.any(np.isnan(model)) or np.any(np.isinf(pixel)) or np.any(np.isnan(pixel)):
            logger.warning('输入数据包含无穷大或NaN值，跳过当前计算')

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

        # 检查L矩阵是否包含无穷大或NaN值
        if np.any(np.isinf(L)) or np.any(np.isnan(L)):
            logger.warning('系数矩阵L包含无穷大或NaN值，返回单位矩阵')

        _, _, Vh = svd(L)
        m = Vh[-1, :]
        assert len(m) == 9  # 可以估计单应性的全部三个行向量
        estimated_homography = m.reshape((3,3))

        # logger.debug(f'估计的单应性 H=\n{estimated_homography}')

        # 检查估计的单应性是否包含无穷大或NaN值
        if np.any(np.isinf(estimated_homography)) or np.any(np.isnan(estimated_homography)):
            logger.warning('估计的单应性包含无穷大或NaN值，返回单位矩阵')

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

        # 防止标准差为0的情况
        std_model = np.where(std_model < 1e-8, 1.0, std_model)
        std_pixel = np.where(std_pixel < 1e-8, 1.0, std_pixel)

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

    @staticmethod
    def approximate_rotation_matrix(matrix_Q: np.ndarray):
        """
        利用基本矩阵计算得到的“旋转矩阵” Q 可能不符合旋转矩阵的正交性（Q^T Q = I），
        因此需要在 F 范数意义下，将它映射为一个“最佳的”旋转矩阵 R 使得 R^T R = I 成立。

        :param matrix_Q: 现有的“旋转矩阵”
        :type matrix_Q: np.ndarray
        """
        U, _, Vt = svd(matrix_Q)
        matrix_R = U @ Vt
        # Ensure det positive
        if (det_R := np.linalg.det(matrix_R)) < 0:
            logger.warning(f'近似旋转矩阵 R=\n{matrix_R} 的行列式 {det_R:.4f} < 0')
            U[:, -1] *= -1
            matrix_R = U @ Vt
        return matrix_R

    @classmethod
    def extract_intrinsic_parameters_from_homography(cls, list_of_homography: typing.List[np.ndarray],
                                                     model_2d_homo: np.ndarray,
                                                     list_of_pixel_2d_homo: typing.List[np.ndarray],
                                                     realK: np.ndarray = None):
        """
        把相机内部参数逐个提取出来
        """

        assert len(list_of_homography) == len(list_of_pixel_2d_homo) > 1, '单应性列表长度应大于零!'

        def v_constraint(homography: np.ndarray, i: int, j: int):
            def h(col: int, row: int):
                # 提取单应性的第 col 列向量的第 row 分量（即单应性的第 row 行第 col 列元素）
                return homography[row-1, col-1]

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

        # # 视图筛选（剔除病态或重投影误差大的视图）
        # logger.debug(f'筛选前有 {len(list_of_homography)} 个视图')
        # list_of_homography_filtered, list_of_pixel_2d_homo_filtered = assert_condition_number(
        #     list_of_homography, list_of_pixel_2d_homo, model_2d_homo,
        #     rmse_threshold=5.0, cond_threshold=1e6
        # )
        # logger.debug(f'筛选前有 {len(list_of_homography_filtered)} 个视图')

        # list_of_homography, list_of_pixel_2d_homo = list_of_homography_filtered, list_of_pixel_2d_homo_filtered

        # 检查V矩阵是否包含无穷大或NaN值
        if np.any(np.isinf(V)) or np.any(np.isnan(V)):
            logger.warning('矩阵V包含无穷大或NaN值!')

        # 计算矩阵 V 的条件数
        # print_all_conditions_of_matrix(V.T @ V, '(V.T @ V)')

        _, S, Vh = svd(V)
        abs_singular_value = abs(S)
        min_abs_singular_value = abs_singular_value.min()
        max_abs_singular_value = abs_singular_value.max()

        if min_abs_singular_value < 1e-12:
            logger.warning('矩阵V的最小奇异值接近0，可能导致数值不稳定')
            # 使用正则化方法求解
            b = Vh[-1, :]
        else:
            logger.debug(f'矩阵 V 的奇异值 = \n\t{ ','.join('{:.6e}'.format(x) for x in S.tolist()) }\n'
                         + f'\t奇异值绝对值最大值 = \t{ max_abs_singular_value :.6e}\n'
                         + f'\t奇异值绝对值最小值 = \t{ min_abs_singular_value :.6e}\n'
                         + f'\t矩阵 V 的谱条件数 = \t{ max_abs_singular_value / min_abs_singular_value :2f}')

            b = Vh[-1, :]

        assert b.shape == (6,), f'向量 b 的形状实际上是: {b.shape}'
        B11, B12, B22, B13, B23, B33 = b
        BB = np.array([
            [B11, B12, B13],
            [B12, B22, B23],
            [B13, B23, B33],
        ], dtype=np.float64)
        logger.debug('估计的基本矩阵(未修改符号) B=\n{BB}')

        # 检查B矩阵元素是否有效
        if np.any(np.isinf(BB)) or np.any(np.isnan(BB)):
            raise ValueError('B矩阵包含无效值，使用默认内参矩阵')

        B11, B12, B22, B13, B23, B33 = b / np.sign(b[0])

        BB = np.array([
            [B11, B12, B13],
            [B12, B22, B23],
            [B13, B23, B33],
        ])
        logger.debug(f'估计的基本矩阵(已修改符号) B=\n{BB}')
        principal_minors = [np.linalg.det(BB[:i, :i]) for i in range(1,4)]
        principal_minors_str = ', '.join(['{:.6e}'.format(x) for x in principal_minors])
        logger.debug(f'估计的基本矩阵的顺序主子式 =\n{principal_minors_str}')
        if not all(pm > 0 for pm in principal_minors):
            logger.warning('估计的基本矩阵不是正定矩阵，可能存在数值不稳定问题！')
            BB = cls.project_to_positive_definite_matrix(BB)
            principal_minors = [np.linalg.det(BB[:i, :i]) for i in range(1,4)]
            principal_minors_str = ', '.join(['{:.6e}'.format(x) for x in principal_minors])
            logger.debug(f'重映射得到的基本矩阵的顺序主子式 =\n{principal_minors_str}')
            B11 = BB[0, 0]
            B12 = BB[0, 1]
            B13 = BB[0, 2]
            B22 = BB[1, 1]
            B23 = BB[1, 2]
            B33 = BB[2, 2]

        mid1 = B12 * B13 - B11 * B23
        mid2 = B11 * B22 - B12 ** 2
        logger.debug(f'{mid1=:.6e}')
        logger.debug(f'{mid2=:.6e}')

        # 检查中间计算值是否有效
        if np.isinf(mid1) or np.isinf(mid2) or np.isnan(mid1) or np.isnan(mid2):
            logger.warning('中间计算值包含无穷大或NaN值，使用默认内参矩阵')

        if abs(mid2) < 1e-12:
            logger.warning('mid2接近0，可能导致数值不稳定，使用默认内参矩阵')

        v0 = mid1 / mid2
        logger.debug(f'{v0=:.6e}')
        rho = B33 - (B13 ** 2 + v0 * mid1) / B11
        alpha0 = rho / B11
        if alpha0 < 0:
            alpha0 = -alpha0
            rho = -rho
            logger.debug('估计的 alpha0 为负数，已取其绝对值进行计算！')
        logger.debug(f'{rho=:.6e}')
        logger.debug(f'{alpha0=:.6e}')
        alpha = np.sqrt(alpha0)
        logger.debug(f'{alpha=:.6e}')
        beta0 = rho * B11 / mid2
        logger.debug(f'{beta0=:.6e}')
        beta = np.sqrt(beta0)
        logger.debug(f'{beta=:.6e}')
        gamma = - B12 * alpha0 * beta / rho
        logger.debug(f'{gamma=:.6e}')
        u0 = gamma * v0 / beta - B13 * alpha0 / rho
        logger.debug(f'{u0=:.6e}')
        K = np.array([
            [alpha, gamma, u0],
            [.0, beta, v0],
            [.0, .0, 1.],
        ], dtype=np.float64)
        logger.debug(f'估计的相机内参矩阵 (initial guess) K=\n{K}')
        if realK is not None:
            evaluate_relative_error(K, realK)

        rvecs_init = []  # 旋转参数
        tvecs_init = []  # 平移参数
        for H in list_of_homography:
            h1, h2, h3 = H.T
            Kinv = np.linalg.inv(K)
            lambda1 = 1.0 / np.linalg.norm(Kinv @ h1)
            lambda2 = 1.0 / np.linalg.norm(Kinv @ h2)

            if abs(lambda1 - lambda2) > 1e-4:
                logger.error(f'尺度因子差距过大: {lambda1=}, {lambda2=}')

            # print(f'>>>>>>>>>>>>>>>>>>> h1 -> ', h1)
            # print(f'>>>>>>>>>>>>>>>>>>> h2 -> ', h2)
            # print(f'>>>>>>>>>>>>>>>>>>> h1 - h2 -> ', h1 - h2)
            # print(f'>>>>>>>>>>>>>>>>>>> Kinv @ h1 -> ', lambda1)
            # print(f'>>>>>>>>>>>>>>>>>>> Kinv @ h2 -> ', lambda2)
            # print(f'>>>>>>>>>>>>>>>>>>> Kinv @ h1 - Kinv @ h2 -> ', lambda1 - lambda2)

            r1 = lambda1 * (Kinv @ h1)
            r2 = lambda1 * (Kinv @ h2)
            t = lambda1 * (Kinv @ h3)
            r3 = np.cross(r1, r2)
            R_approx = np.column_stack([r1, r2, r3])
            R = cls.approximate_rotation_matrix(R_approx)
            rvec = inv_rodrigues(R)
            rvecs_init.append(rvec)
            tvecs_init.append(t)

        # 用牛顿法

        def pack_params(K: np.ndarray, rvecs: list[np.ndarray], tvecs: list[np.ndarray]) -> np.ndarray:
            """
            将相机内参矩阵、旋转参数、平移参数全部打包为一个向量
            """
            parts = [
                # 只优化相机内参矩阵的部分参数（共计 5 个参数）
                [K[0], K[1, 1:]],
                (rvec.reshape(-1) for rvec in rvecs),
                (tvec.reshape(-1) for tvec in tvecs),
            ]
            return np.concatenate(list(itertools.chain.from_iterable(parts))).astype(np.float64)

        def unpack_params(x: np.ndarray, n_views: int):
            """
            从打包好的向量中拆出相机内参矩阵、旋转参数、平移参数
            """
            n_params_in_K = 5
            x_in_K = x[:n_params_in_K]  # 首先提取相机内部参数
            K = np.array([  # 构造相机内参矩阵
                [x_in_K[0], x_in_K[1], x_in_K[2]],
                [0.0, x_in_K[3], x_in_K[4]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            rest = x[n_params_in_K:]  # 剩余的参数
            len_rest = rest.shape[0]
            assert len_rest % 2 == 0, '剩余参数个数应该是偶数!'
            assert len_rest == (len_rest_expected := 2 * n_views * 3), \
                f'实际剩余参数个数 ({len_rest}) 与预期 ({len_rest_expected}) 不同!'
            len_half_rest = len_rest // 2
            rvecs = rest[:len_half_rest]
            tvecs = rest[len_half_rest:]
            rvecs = rvecs.reshape((n_views, 3))
            tvecs = tvecs.reshape((n_views, 3))
            return K, rvecs, tvecs

        n_views = len(list_of_pixel_2d_homo)  # 视图个数（即不同姿态的相机拍摄的“照片”的张数）
        n_iter = 0

        def residuals_joint(x: np.ndarray) -> np.ndarray:
            K, rvecs, tvecs = unpack_params(x, n_views)
            K = K.astype(np.float64)
            K /= K[2, 2]  # 相机内参矩阵归一化
            residuals = []

            assert model_2d_homo.ndim == 2 and model_2d_homo.shape[1] == 3, \
                f'{model_2d_homo.ndim=}, {model_2d_homo.shape=}'

            for rv, tv, pixel_2d_homo in zip(rvecs, tvecs, list_of_pixel_2d_homo):
                tv = tv.reshape(3, 1)
                R = rodrigues(rv)  # 利用旋转参数 rv 构造旋转矩阵 R。
                H = CameraModel.make_homography(K, R, tv)  # 利用相机内参矩阵 K、旋转矩阵 R 和平移向量 tv 构造单应性 H。
                assert H.shape == (3, 3), f'{H.shape=}'
                reprojection_points_homo = model_2d_homo @ H.T  # 重投影，产出像素点的齐次坐标
                reprojection_points_nonhomo = homo2nonhomo(reprojection_points_homo)
                pixel_2d_nonhomo = homo2nonhomo(pixel_2d_homo)
                diff = (reprojection_points_nonhomo - pixel_2d_nonhomo).reshape(-1)
                residuals.append(diff)

            nonlocal n_iter
            if n_iter % 100 == 0:  # 每 100 次迭代，绘图一次
                path_fig = f'fig-0-optimizition (iter {n_iter:04d}).png'
                CameraModel.visualize_reprojection(model_2d_homo, list_of_pixel_2d_homo, K, rvecs, tvecs, 0, path_fig)
            n_iter += 1

            return np.concatenate(residuals).astype(np.float64)

        def homography_reprojection_rmse(x: np.ndarray) -> float:
            err_vec = residuals_joint(x).reshape((-1, 2))
            err_norm2 = np.linalg.norm(err_vec, axis=1) ** 2
            err_mean = np.mean(err_norm2)
            return np.sqrt(err_mean)

        x0 = pack_params(K, rvecs_init, tvecs_init)

        logger.debug(f'优化之前的重投影误差：\n\t{homography_reprojection_rmse(x0)}')

        n_params = len(x0)  # 需要调优的参数个数
        lower_bounds = -np.inf * np.ones(n_params)
        upper_bounds = np.inf * np.ones(n_params)
        # 强制 alpha > 0 (给一个极小正数防止除0)
        lower_bounds[0] = 1e-9
        # 强制 beta > 0 (注意 pack_params 中: K[0]有3个元素，K[1,1:]有2个元素，beta是第4个元素，索引为3)
        lower_bounds[3] = 1e-9

        optimize_result: scipy.optimize.OptimizeResult = scipy.optimize.least_squares(
            residuals_joint,
            x0=x0,
            # method='lm',  # <-- Levenberg-Marquardt algorithm 不支持 bounds，必须注释掉
            method='trf',   # <-- 改用 Trust Region Reflective 算法，支持边界约束
            bounds=(lower_bounds, upper_bounds), # <-- 传入边界
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=5000,  # 最多迭代 5000 * n 次
            verbose=2,
        )

        # print(f'{optimize_result=}')
        x_opt: np.ndarray = optimize_result.x
        K_opt, rvecs_opt, tvecs_opt = unpack_params(x_opt, n_views)
        K_opt /= K_opt[2, 2]  # 相机内参矩阵归一化

        logger.debug(f'优化之后的重投影误差：\n\t{homography_reprojection_rmse(x_opt)}')

        for idx in range(len(list_of_pixel_2d_homo)):
            CameraModel.visualize_reprojection(model_2d_homo, list_of_pixel_2d_homo, K_opt, rvecs_opt, tvecs_opt, idx)

        return K_opt # / K_opt[2, 2]



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


def init():
    camera_theta = np.radians(90)
    # projection_model = CameraModel(
    #     d=100,
    #     a=400,
    #     b=300,
    #     theta=np.radians(90.5),  # 将角度制的 90.5° 转为弧度制
    #     u0=320,
    #     v0=240,
    # )
    projection_model = CameraModel(
        d=100,
        a=100,
        b=100,
        theta=camera_theta,  # 将角度制的 90.5° 转为弧度制
        u0=0,
        v0=0,
    )
    model_points = generate_model_points()
    n_photos = 20

    list_of_image_points = []
    list_of_rotation = []
    list_of_translation = []

    for _ in range(n_photos):
        _, image_points, rotation, translation, _ = projection_model.randomly_project(model_points)
        list_of_image_points.append(image_points)
        list_of_rotation.append(rotation)
        list_of_translation.append(translation)


    def infer_image_size(margin: int = 2, min_size: tuple[int, int] = (480, 640)):
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

        # # 检查数据是否包含无穷大或NaN值
        # if np.any(np.isinf(arr)) or np.any(np.isnan(arr)):
        #     logger.warning('输入图像点包含无穷大或NaN值，将使用默认尺寸')
        #     return min_size[0], min_size[1]

        # 把齐次坐标除以第三个分量得到非齐次 u,v
        # 防止除以0 (理论上第三分量都是1)，用 eps 保护
        denom = arr[..., 2:3]
        # 使用更安全的除法，防止除以接近0的值
        denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
        uv = arr[..., :2] / denom   # shape (V, M, 2)

        # 检查uv中是否包含无穷大或NaN值
        if np.any(np.isinf(uv)) or np.any(np.isnan(uv)):
            logger.warning('计算得到的uv坐标包含无穷大或NaN值，将使用默认尺寸')
            return min_size[0], min_size[1]

        # 取所有视图与所有点的最大 u (x) 和 v (y)
        max_u = np.nanmax(uv[..., 0])
        max_v = np.nanmax(uv[..., 1])

        # 检查最大值是否为无穷大
        if np.isinf(max_u) or np.isinf(max_v):
            logger.warning('最大坐标值为无穷大，将使用默认尺寸')
            return min_size[0], min_size[1]

        width  = int(np.ceil(max_u)) + int(margin)
        height = int(np.ceil(max_v)) + int(margin)

        # 应用最小尺寸下限
        height = max(height, int(min_size[0]))
        width  = max(width,  int(min_size[1]))

        return (height, width)

    image_size = infer_image_size()

    save_mat(
        'zhang.mat',
        model_points,           # 真实的模型点二维齐次坐标
        image_size,
        list_of_image_points,
        projection_model.K,     # 真实的相机内参矩阵
        list_of_rotation,       # 真实的旋转矩阵
        list_of_translation,    # 真实的平移向量
    )


def show_real_homography(list_of_rotation, list_of_translation, real_K):
    real_homography = [
        CameraModel.make_homography(real_K, rotation, translation)
        for rotation, translation in zip(list_of_rotation, list_of_translation)
    ]
    for homography in real_homography:
        h_norm = homography / homography[2, 2]
        print(f'正在检查单应性 =\n {h_norm}')
        h_cond = np.linalg.cond(h_norm)
        print(f'真实单应性条件数 = {h_cond}')


def evaluate_relative_error(estimated_intrinsic_matrix, real_intrinsic_matrix):
    tiny = np.finfo(real_intrinsic_matrix.dtype).tiny
    relative_error_norm = np.linalg.norm(estimated_intrinsic_matrix - real_intrinsic_matrix) / (np.linalg.norm(real_intrinsic_matrix) + tiny)
    logger.info(f'相机内参矩阵相对误差(L2范数) = {relative_error_norm*100:.2f}%')
    sign = np.sign(real_intrinsic_matrix)
    signed_tiny = np.where(sign == 0.0, tiny, tiny * sign)
    relative_error_elementwise = (estimated_intrinsic_matrix - real_intrinsic_matrix) / (real_intrinsic_matrix + signed_tiny)
    relative_error_elementwise[relative_error_elementwise > 1e10] = np.inf
    relative_error_elementwise[relative_error_elementwise < -1e10] = -np.inf
    formatted_array = np.array2string(
        relative_error_elementwise * 100,
        formatter={'float_kind': lambda x: f'{x:.2f}%'},
        precision=2,
        suppress_small=True,
        separator=', '
    )
    logger.debug(f'相机内参矩阵相对误差(逐个元素) =\n{formatted_array}')


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
            uv = homo2nonhomo(pts)
        # 也可能是 (M,2)
        elif pts.ndim == 2 and pts.shape[1] == 2:
            uv = pts
        # 也可能是 (1, M, 3) 或 (M, 3, 1) 等，尝试 reshape
        else:
            pts_flat = pts.reshape(-1, pts.shape[-1])
            if pts_flat.shape[1] == 3:
                uv = homo2nonhomo(pts_flat)
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
    logger.info(f'OpenCV 总重投影 RMSE = {rmse:.6e} 像素')

    # 与真实内参比较（如果存在）
    realK = saved_data.get('real_intrinsic_matrix')
    if realK is not None:
        try:
            realK = np.asarray(realK, dtype=np.float32)
            # 如果 loadmat 导致维度多一级，扁平化
            if realK.shape != (3,3):
                realK = realK.reshape(3,3)
            evaluate_relative_error(camera_matrix, realK)
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


def min_max(iterable, dtype=np.float32):
    min_val = dtype('inf')
    max_val = dtype('-inf')
    # 在迭代过程中直接更新
    for value in iterable:
        if value < min_val:
            min_val = value
        if value > max_val:
            max_val = value
    return min_val, max_val


def print_homography_condition(list_of_homography: list[np.ndarray]):
    min_cond_homography, max_cond_homography = min_max(
        np.linalg.cond(homography)
        for homography in list_of_homography
    )
    logger.debug(f'经过归一化以后的单应性的条件数，最小值 = {min_cond_homography:.6e}，最大值 = {max_cond_homography:.6e}')


def assert_quasi_affine(list_of_homography: list[np.ndarray], model_points: np.ndarray) -> list[int]:
    """
    返回被判定为 quasi-affine 的视图索引列表（indices），以及按这些索引筛选后的 homographies。
    这样可以确保后续用到的图像点集合与单应性列表一一对应。
    """
    kept_idx = []
    for idx, homography in enumerate(list_of_homography):
        try:
            h_inv = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            continue
        h_inv_r3 = h_inv[2, :]
        seq = model_points @ h_inv_r3.T
        seq = np.sign(seq)
        rate = abs(seq.sum()) / len(seq)
        # logger.debug(f'单应性拟仿射性 = {rate}')
        if rate > .95:
            kept_idx.append(idx)
    remain_ratio = len(kept_idx) / max(1, len(list_of_homography))
    logger.debug(f'拟仿射筛选剩余率 = {remain_ratio * 100:.2f}%')
    return kept_idx


def run():
    saved_data = load_mat('zhang.mat')
    model_points = saved_data['model_2d_homo']
    list_of_image_points = saved_data['list_of_pixel_2d_homo']

    for idx in range(len(list_of_image_points)):
        CameraModel.visualize_projection(model_points, list_of_image_points, idx)

    realK = saved_data['real_intrinsic_matrix']
    logger.debug(f'可用校正图像数量: {len(list_of_image_points)}')
    with Timer():
        list_of_homography = []
        # 为各个不同视图分别计算单应性
        for image_points in list_of_image_points:
            homography = ZhangCameraCalibration.infer_homography_without_radial_distortion_with_isotropic_scaling(
                model_points, image_points
            )
            list_of_homography.append(homography)
        # 利用单应性的 quasi-affine 假设，进行筛选
        kept_idx = assert_quasi_affine(list_of_homography, model_points)
        # kept_idx = kept_idx[:10]
        list_of_homography_filtered = [
            list_of_homography[i]
            for i in kept_idx
        ]
        list_of_image_points_filtered = [
            list_of_image_points[i]
            for i in kept_idx
        ]
        # 打印单应性的条件数
        print_homography_condition(list_of_homography_filtered)
        # 估计相机内参
        K = ZhangCameraCalibration.extract_intrinsic_parameters_from_homography(
            list_of_homography_filtered, model_points, list_of_image_points_filtered,
            realK=realK,
        )
    logger.debug(f'估计的相机内参矩阵 K=\n{K}')
    evaluate_relative_error(K, realK)


if __name__ == '__main__':
    init()  # 生成模型点和像素点

    logger.debug('\n' * 10 + '=' * 100)
    saved_data = load_mat('zhang.mat')
    image_size = saved_data['image_size']
    logger.debug(f'图像尺寸 ={image_size}')
    realK = saved_data['real_intrinsic_matrix']
    realKinv = np.linalg.inv(realK)
    logger.debug(f'真实的相机内参矩阵 K=\n{realK}')
    logger.debug(f'真实的基本矩阵 B=\n{realKinv.T @ realKinv}')

    def delete_all_pngs():
        # 获取当前目录对象
        current_dir = pathlib.Path('.')

        # 查找所有 .png 文件 (不区分大小写通常取决于操作系统，Linux下区分)
        # 如果想同时匹配 .PNG 和 .png，可以使用 glob('*.[pP][nN][gG]')
        png_files = list(current_dir.glob('*.png'))

        if not png_files:
            print("当前目录下没有找到 PNG 文件。")
            return

        print(f"正在删除 {len(png_files)} 个文件...")

        for file_path in png_files:
            try:
                file_path.unlink()  # 执行删除
                print(f"已删除: {file_path.name}")
            except Exception as e:
                print(f"删除失败 {file_path.name}: {e}")

    delete_all_pngs()

    logger.debug('\n' * 10 + '=' * 100)

    run()

    # 使用 opencv 现有的算法，求解相机内参矩阵
    compare_with_opencv()
