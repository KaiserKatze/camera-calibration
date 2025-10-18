#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import typing

import numpy as np

type ListOf3DPoints = typing.List[typing.Tuple[float, float, float, float]]
type ListOf2DPoints = typing.List[typing.Tuple[float, float]]


np.set_printoptions(linewidth=np.inf)


class Model:
    def __init__(self, world_coordinates: ListOf3DPoints):
        self.world_coordinates = np.array(world_coordinates)


class Pixel:
    def __init__(self, pixel_coordinates: ListOf2DPoints):
        self.pixel_coordinates = np.array(pixel_coordinates)


class Calibrator:
    def __init__(self, model: Model, pixel: Pixel):
        self.model = model.world_coordinates
        self.pixel = pixel.pixel_coordinates
        # assert len(self.model) == len(self.pixel) > 6
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
        T = rho * (np.linalg.inv(K) @ b)
        # print(f'{r1=}')
        # print(f'{r2=}')
        # print(f'{r3=}')
        print('相机外参数：')
        print(f'{R=}')
        print(f'{T=}')


if __name__ == '__main__':
    A = [
        (1,2),
        (3,4),
    ]

    B = [
        (1,2,3,1),
        (3,4,5,1),
    ]

    Calibrator(Model(B), Pixel(A))
