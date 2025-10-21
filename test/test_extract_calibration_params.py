#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import typing

import numpy as np
import scipy.optimize


np.set_printoptions(linewidth=np.inf)


class NotCalibrator:
    def __init__(self, model: np.ndarray, pixel: np.ndarray):
        model_h, model_w = model.shape
        pixel_h, pixel_w = pixel.shape
        assert model_h == pixel_h, f'模型点、像点数量不相等: 模型点数量={model_h}, 像点数量={pixel_h}'
        assert pixel_h > 6, f'样本点数量太少: {pixel_h}'
        assert model_w == 4, '必须输入模型的齐次坐标'
        assert pixel_w == 2, '必须输入影像的非齐次坐标'
        self.model = model
        self.pixel = pixel

        z = np.zeros(4)
        P = np.block(
            list(
                itertools.chain.from_iterable(
                    [
                        [model_point.T, z, -pixel_point[0] * model_point.T],
                        [z, model_point.T, -pixel_point[1] * model_point.T],
                    ]
                    for model_point, pixel_point in zip(self.model, self.pixel)
                )
            )
        )
        _, _, V = np.linalg.svd(P, full_matrices=True, compute_uv=True, hermitian=False)
        m = V[:, -1]
        assert len(m) == 12
        M = m.reshape((3, 4))
        A = M[:, 0:3]  # 取出矩阵 M 的前三列
        b = M[:, 3]  # 取出矩阵 M 的第四列（也是最后一列）
        a1, a2, a3 = A  # 取出行向量 a1, a2, a3
        # print(f'{P=}')
        # print(f'{V=}')
        # print(f'{m=}')
        # print(f'{M=}')
        # print(f'{A=}')
        # print(f'{b=}')
        # norm_m = np.linalg.norm(m)
        # print(f'{norm_m=}')
        # s = P @ m
        # print(f'{s=}')
        # print(f'{a1=}')
        # print(f'{a2=}')
        # print(f'{a3=}')
        a3_norm = np.linalg.norm(a3)
        rho = 1 / a3_norm
        rho2 = rho * rho
        u0 = rho2 * np.dot(a1, a3)
        v0 = rho2 * np.dot(a2, a3)
        a1_cross_a3 = np.cross(a1, a3)
        a2_cross_a3 = np.cross(a2, a3)
        a1_cross_a3_norm = np.linalg.norm(a1_cross_a3)
        a2_cross_a3_norm = np.linalg.norm(a2_cross_a3)
        # alpha_over_sin_theta = rho2 * a1_cross_a3_norm
        beta_over_sin_theta = rho2 * a2_cross_a3_norm
        cos_theta = - np.dot(a1_cross_a3, a2_cross_a3) / (a1_cross_a3_norm * a2_cross_a3_norm)
        sin_theta = np.sqrt(1 - cos_theta * cos_theta)
        theta = np.arccos(cos_theta)
        alpha = rho2 * a1_cross_a3_norm * sin_theta
        beta = rho2 * a2_cross_a3_norm * sin_theta
        K = np.array(
            [
                [alpha, - rho2 * a1_cross_a3_norm * cos_theta, u0],
                [0, beta_over_sin_theta, v0],
                [0, 0, 1],
            ]
        )
        # print(f'{a1_cross_a3=}')
        # print(f'{a2_cross_a3=}')
        # print(f'{cos_theta=}')
        # print(f'{sin_theta=}')
        # print(f'{alpha_over_sin_theta=}')
        # print(f'{beta_over_sin_theta=}')

        print('相机内参数：')
        # print(f'{alpha=}')
        # print(f'{beta=}')
        # print(f'{theta=}')
        # print(f'{u0=}')
        # print(f'{v0=}')
        # print(f'{rho=}')
        print(f'{K=}')

        r1 = a2_cross_a3 / a2_cross_a3_norm
        r3 = a3 / a3_norm
        r2 = np.cross(r3, r1)
        R = np.concat([r1, r2, r3]).reshape((3, 3))
        T = rho * (np.linalg.pinv(K) @ b)
        # print(f'{r1=}')
        # print(f'{r2=}')
        # print(f'{r3=}')
        print('相机外参数：')
        print(f'{R=}')
        print(f'{T=}')


class RadialCalibrator:
    @staticmethod
    def make_matrix_L(model: np.ndarray, pixel: np.ndarray):
        model_h, model_w = model.shape
        pixel1 = pixel @ np.array([
            [0, 1],
            [-1, 0],
        ])
        out = np.empty((model_h, 2 * model_w), dtype=np.result_type(model.dtype, pixel.dtype))
        np.multiply(model, pixel1[:, :1], out=out[:, :model_w])
        np.multiply(model, pixel1[:, 1:2], out=out[:, model_w:])
        return out

    def __init__(self, model: np.ndarray, pixel: np.ndarray):
        model_h, model_w = model.shape
        pixel_h, pixel_w = pixel.shape
        assert model_h == pixel_h, f'模型点、像点数量不相等: 模型点数量={model_h}, 像点数量={pixel_h}'
        assert pixel_h > 6, f'样本点数量太少: {pixel_h}'
        assert model_w == 4, '必须输入模型的齐次坐标'
        assert pixel_w == 2, '必须输入影像的非齐次坐标'

        L = self.make_matrix_L(model, pixel)
        _, _, V = np.linalg.svd(L, full_matrices=True, compute_uv=True, hermitian=False)
        m = V[:, -1]
        assert len(m) == 8
        M_without_m3 = m.reshape((2, 4))  # 只有 m1, m2, 没有 m3
        # print(f'{L=}')
        print(f'{M_without_m3=}')


if __name__ == '__main__':
    # 定义常量参数
    m = 8  # 网格行数
    n = 8  # 网格列数
    a = 25  # 每个黑色正方形的边长（像素）
    b = 30  # 相邻正方形之间的间距（像素）
    c = 50  # 正方形与图片边缘的最小距离（像素）

    model = []
    for i in range(m):
        for j in range(n):
            # 计算当前正方形的左上角坐标
            x_start = c + j * (a + b)
            y_start = c + i * (a + b)

            # 计算当前正方形的右下角坐标
            x_end = x_start + a
            y_end = y_start + a

            model.extend([
                (x_start, y_start, 0, 1),
                (x_start, y_end, 0, 1),
                (x_end, y_start, 0, 1),
                (x_end, y_end, 0, 1),
            ])
    model = np.array(model, dtype=np.float64)
    pixel = model[:, 0:2]

    RadialCalibrator(model, pixel)
