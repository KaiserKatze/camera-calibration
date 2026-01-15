#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
calibrate_with_opencv.py

读取 model.npy 和 pixel.npy，自动解析为每视图的 object/image points，
使用 OpenCV 的 calibrateCamera 做相机标定并打印内参矩阵。

用法:
    python calibrate_with_opencv.py

可选: 如果你知道每视图的角点数（points_per_view），可以在 main() 中设置。
"""

import numpy as np
import cv2 as cv
import os
from typing import List, Tuple, Optional, Dict


def load_npy(path: str) -> np.ndarray:
    """安全加载 .npy 文件（allow_pickle=False 时保存的文件）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return np.load(path, allow_pickle=False)


def infer_views_from_arrays(model: np.ndarray,
                            pixel: np.ndarray,
                            points_per_view: Optional[int] = None,
                            tol: float = 1e-8
                            ) -> List[Dict[str, np.ndarray]]:
    """
    将 model 与 pixel 转为 views 列表。
    返回 values: [{'world': Nx2_or_... , 'image': Nx2}, ...]
    支持的输入形式 (典型来自你的 Python 保存方式):
      - model: (N,2)  & pixel: (N,2)  -> single view (unless concatenation detected)
      - model: (V,N,2) & pixel: (V,N,2)
      - model: (N,2,V) & pixel: (N,2,V)
      - model: (V*N,2) & pixel: (V*N,2) where model blocks are repeated -> will detect and split
      - model may be (M,3) or (M,4) (homogeneous) -> first two columns taken as X,Y
    如果 points_per_view 指定，会直接按该值拆分（更可靠）。
    """
    if model.shape[0] != pixel.shape[0] and model.ndim == 2 and pixel.ndim == 2:
        # shapes mismatch in rows could still be ok for (V,N,2) vs (N,2,V) etc.
        pass

    # Normalize model to a Nx2 array if it's Nx3 or Nx4 (homogeneous), take first 2 cols.
    if model.ndim == 2 and model.shape[1] >= 2:
        if model.shape[1] > 2:
            # If 3 or 4 columns (possibly homogeneous), take first two columns as plan coordinates X,Y
            model2 = model[:, :2].astype(np.float64)
        else:
            model2 = model.astype(np.float64)
    else:
        model2 = model

    pixel2 = pixel.astype(np.float64)

    # Case: 3D arrays
    if model2.ndim == 3 and pixel2.ndim == 3:
        sm = model2.shape
        sp = pixel2.shape
        # (V,N,2)
        if sm[2] == 2 and sp[2] == 2 and sm[0] == sp[0] and sm[1] == sp[1]:
            V, N, _ = sm
            views = []
            for v in range(V):
                views.append({'world': model2[v, :, :].astype(np.float32),
                              'image': pixel2[v, :, :].astype(np.float32)})
            return views
        # (N,2,V)
        if sm[1] == 2 and sp[1] == 2 and sm[2] == sp[2]:
            N, _, V = sm
            views = []
            for v in range(V):
                views.append({'world': model2[:, :, v].astype(np.float32),
                              'image': pixel2[:, :, v].astype(np.float32)})
            return views
        raise ValueError("Unsupported 3D shape combination for model/pixel.")

    # Case: 2D arrays (M,2) possibly concatenated across views
    if model2.ndim == 2 and pixel2.ndim == 2:
        M, C = model2.shape
        if C != 2 or pixel2.shape[1] != 2:
            raise ValueError("Expect 2 columns for coordinates (x,y) in model and pixel arrays.")

        # If user provided points_per_view, do straightforward split
        if points_per_view is not None:
            N = int(points_per_view)
            if M % N != 0:
                raise ValueError(f"points_per_view={N} does not divide total points M={M}")
            V = M // N
            views = []
            for v in range(V):
                s = v * N
                e = (v + 1) * N
                views.append({'world': model2[s:e, :].astype(np.float32),
                              'image': pixel2[s:e, :].astype(np.float32)})
            print(f"[info] split by provided points_per_view: V={V}, N={N}")
            return views

        # 自动检测：查找能把 M 分成若干块，每块与第一块相同的 N
        def divisors(n):
            ds = []
            for i in range(1, int(np.sqrt(n)) + 1):
                if n % i == 0:
                    ds.append(i)
                    if i != n // i:
                        ds.append(n // i)
            return sorted(ds)

        ds = divisors(M)
        # candidate N: exclude 1 and M itself, require N>=4
        cand = [d for d in ds if 4 <= d < M]
        for N in cand:
            V = M // N
            ok = True
            base = model2[0:N, :]
            for v in range(1, V):
                block = model2[v * N:(v + 1) * N, :]
                if not np.allclose(block, base, atol=tol, rtol=0):
                    ok = False
                    break
            if ok:
                views = []
                for v in range(V):
                    s = v * N
                    e = (v + 1) * N
                    views.append({'world': model2[s:e, :].astype(np.float32),
                                  'image': pixel2[s:e, :].astype(np.float32)})
                print(f"[detect] Detected concatenation: M={M} => V={V} views, N={N} points/view")
                return views

        # fallback: treat as single view
        print("[detect] No repetition pattern detected; treating arrays as single view.")
        return [{'world': model2.astype(np.float32), 'image': pixel2.astype(np.float32)}]

    raise ValueError("Unsupported array dimensionality.")


def prepare_for_opencv(views: List[Dict[str, np.ndarray]]) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    """
    把 views 转成 OpenCV 所需的 objpoints, imgpoints 列表，以及估计 imageSize。
    objpoints: list of (N,1,3) arrays (float32)
    imgpoints: list of (N,1,2) arrays (float32)
    image_size: (width, height) estimated from max pixel coordinates (as integers)
    """
    objpoints = []
    imgpoints = []
    max_u = 0.0
    max_v = 0.0
    for v in views:
        w = v['world']  # Nx2 or Nx? take first two cols
        img = v['image']  # Nx2
        if w.shape[1] < 2:
            raise ValueError("world points must have at least 2 columns (X,Y)")
        # make 3D points with Z=0
        pts3d = np.zeros((w.shape[0], 3), dtype=np.float32)
        pts3d[:, 0:2] = w[:, 0:2].astype(np.float32)
        objpoints.append(pts3d.reshape(-1, 1, 3))
        imgpoints.append(img.astype(np.float32).reshape(-1, 1, 2))
        max_u = max(max_u, float(np.max(img[:, 0])))
        max_v = max(max_v, float(np.max(img[:, 1])))

    # estimate image size as (width, height) at least 1 pixel bigger than max coordinate
    # if your original images have known size, you should pass that size instead
    image_size = (int(np.ceil(max_u)) + 1, int(np.ceil(max_v)) + 1)
    return objpoints, imgpoints, image_size


def calibrate_and_report(objpoints: List[np.ndarray],
                         imgpoints: List[np.ndarray],
                         image_size: Tuple[int, int]):
    """
    调用 OpenCV 的 calibrateCamera，打印结果并返回 (ret, K, dist, rvecs, tvecs).
    还计算并打印重投影误差 RMS。
    """
    # flags: let OpenCV estimate skew if present; you can set flags=cv.CALIB_ZERO_TANGENT_DIST etc.
    flags = 0  # 默认：估计全部（包含畸变）
    # You can set e.g. flags = cv.CALIB_FIX_K3 to fix third distortion term, etc.

    # termination criteria for refine (optional)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 1e-6)

    # run calibrateCamera
    ret, K, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=criteria
    )

    print("=== calibrateCamera results ===")
    print(f"RMS re-projection error reported by OpenCV: {ret}")
    print("Camera intrinsic matrix K:\n", K)
    print("Distortion coefficients (k1,k2,p1,p2,k3,...):\n", distCoeffs.ravel())

    # compute own reprojection error per-point and global RMS
    total_err = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, distCoeffs)
        # imgpoints2 shape: (N,1,2)
        err = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)
        n = objpoints[i].shape[0]
        total_err += err**2
        total_points += n
    rms = np.sqrt(total_err / total_points)
    print(f"Computed global RMS reprojection error: {rms:.6f} pixels")
    return ret, K, distCoeffs, rvecs, tvecs


def visualize_reprojection(objpoints: List[np.ndarray],
                           imgpoints: List[np.ndarray],
                           K,
                           distCoeffs,
                           rvecs,
                           tvecs,
                           view_index: int = 0):
    """
    可选：可视化第 view_index 个视图的原始点与重投影点 (需要显示环境)。
    """
    import matplotlib.pyplot as plt
    obj = objpoints[view_index]
    img_obs = imgpoints[view_index].reshape(-1, 2)
    img_proj, _ = cv.projectPoints(obj, rvecs[view_index], tvecs[view_index], K, distCoeffs)
    img_proj = img_proj.reshape(-1, 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(img_obs[:, 0], img_obs[:, 1], c='r', marker='o', label='observed')
    plt.scatter(img_proj[:, 0], img_proj[:, 1], c='b', marker='x', label='projected')
    for i in range(img_obs.shape[0]):
        plt.plot([img_obs[i, 0], img_proj[i, 0]], [img_obs[i, 1], img_proj[i, 1]], 'g-', linewidth=0.5)
    plt.gca().invert_yaxis()  # 图像坐标通常原点在左上
    plt.legend()
    plt.title(f"View {view_index} reprojection (red:obs, blue:proj)")
    plt.show()


def main():
    # 修改为你的文件名或路径（默认与脚本同目录）
    model_path = "model.npy"
    pixel_path = "pixel.npy"

    # 如果你知道每视图点数（例如 256），把它写在这里以避免自动检测错误
    points_per_view = None  # e.g. 256

    model = load_npy(model_path)
    pixel = load_npy(pixel_path)
    print("Loaded shapes: model", model.shape, "pixel", pixel.shape)

    views = infer_views_from_arrays(model, pixel, points_per_view=points_per_view)
    print(f"Parsed into {len(views)} views.")

    objpoints, imgpoints, image_size = prepare_for_opencv(views)
    print("Estimated image size (used for calibrateCamera):", image_size)

    ret, K, distCoeffs, rvecs, tvecs = calibrate_and_report(objpoints, imgpoints, image_size)

    # 若想可视化第一个视图的重投影（需要 matplotlib）
    try:
        visualize_reprojection(objpoints, imgpoints, K, distCoeffs, rvecs, tvecs, view_index=0)
    except Exception as e:
        print("Visualization skipped (matplotlib may be missing or not available).", e)


if __name__ == "__main__":
    main()
