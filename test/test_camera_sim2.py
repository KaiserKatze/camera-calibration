#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import json
import os
import traceback


# -------------------------
# GPU 上下文与包装
# -------------------------
class GpuContext:
    def __init__(self):
        self.available = False
        self.device_count = 0
        self.errmsg = ""
        self._detect()

    def _detect(self):
        try:
            if not hasattr(cv2, "cuda"):
                self.errmsg = "OpenCV 未包含 cuda 模块。"
                return
            self.device_count = int(cv2.cuda.getCudaEnabledDeviceCount())
            if self.device_count <= 0:
                self.errmsg = "未检测到可用 CUDA 设备。"
                return
            # 尝试设置设备 & 分配 GpuMat 以确认可用
            cv2.cuda.setDevice(0)
            _ = cv2.cuda_GpuMat()
            self.available = True
        except Exception as e:
            self.errmsg = f"CUDA 初始化失败：{e}"

    def upload(self, img_np):
        try:
            g = cv2.cuda_GpuMat()
            g.upload(img_np)
            return g
        except Exception:
            return None

    def download(self, gmat):
        try:
            return gmat.download()
        except Exception:
            return None

    def warp_perspective(self, g_src, H, size, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)):
        try:
            # OpenCV CUDA API: cv2.cuda.warpPerspective(src, M, dsize, flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue)
            g_dst = cv2.cuda.warpPerspective(
                g_src, H, size,
                flags=cv2.INTER_LINEAR,
                borderMode=borderMode,
                borderValue=borderValue
            )
            return g_dst
        except Exception:
            # 可能某些 OpenCV 版本不支持 borderValue，尝试不传递该参数
            try:
                g_dst = cv2.cuda.warpPerspective(
                    g_src, H, size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=borderMode
                )
                return g_dst
            except Exception:
                return None

    def warp_affine(self, g_src, M, size, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)):
        try:
            g_dst = cv2.cuda.warpAffine(
                g_src, M, size,
                flags=cv2.INTER_LINEAR,
                borderMode=borderMode,
                borderValue=borderValue
            )
            return g_dst
        except Exception:
            try:
                g_dst = cv2.cuda.warpAffine(
                    g_src, M, size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=borderMode
                )
                return g_dst
            except Exception:
                return None

    def remap(self, g_src, g_mapx, g_mapy, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)):
        try:
            g_dst = cv2.cuda.remap(
                g_src, g_mapx, g_mapy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=borderMode,
                borderValue=borderValue
            )
            return g_dst
        except Exception:
            try:
                g_dst = cv2.cuda.remap(
                    g_src, g_mapx, g_mapy,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=borderMode
                )
                return g_dst
            except Exception:
                return None

    def split(self, g_src):
        try:
            return cv2.cuda.split(g_src)
        except Exception:
            return None

    def merge(self, g_channels):
        try:
            return cv2.cuda.merge(g_channels)
        except Exception:
            return None

    def linear_filter(self, g_src, kernel):
        # 运动模糊卷积（线性滤波）GPU 版（若不可用则返回 None）
        try:
            k = kernel.astype(np.float32)
            g_kernel = cv2.cuda_GpuMat()
            g_kernel.upload(k)
            # 某些版本需要通过 createLinearFilter 创建
            # 兼容：优先使用 createLinearFilter
            try:
                filt = cv2.cuda.createLinearFilter(g_src.type(), g_src.type(), k)
                return filt.apply(g_src)
            except Exception:
                # 直接 filter2D 可能不可用，返回 None 交给 CPU
                return None
        except Exception:
            return None

    def multiply(self, g_a, g_b):
        try:
            return cv2.cuda.multiply(g_a, g_b)
        except Exception:
            return None


class CameraSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("相机透视与畸变模拟器（含缺陷模拟）")
        self.root.geometry("1400x820")

        # GPU 上下文
        self.gpu = GpuContext()
        if self.gpu.available:
            print(f"[INFO] 检测到 CUDA 设备数: {self.gpu.device_count}，已启用 GPU 加速。")
        else:
            print(f"[INFO] 未启用 GPU 加速：{self.gpu.errmsg}")

        # 加载默认图像
        self.original_image = cv2.imread('square_grid.png')
        if self.original_image is None:
            self.original_image = self.create_checkerboard_image()

        # 预览缩放缓存
        self.preview_image = None
        self.photo_image = None

        # 更新防抖
        self.update_job = None

        # 图像尺寸变化时，缓存需要重建
        self._invalidate_all_caches()

        # 参数（默认值）
        self.default_params = {
            # 几何（影响单应性）—— 缓存1：warped
            'translation_x': 0.0,
            'translation_y': 0.0,
            'rotation_yaw': 0.0,
            'rotation_pitch': 0.0,
            'rotation_roll': 0.0,
            'fov_deg': 60.0,

            # 畸变（光学）—— 缓存2：lens
            'k1': 0.0,
            'k2': 0.0,
            'k3': 0.0,
            'p1': 0.0,
            'p2': 0.0,

            # 缺陷/后效
            'vign_strength': 0.0,   # 0~1
            'vign_exponent': 2.0,   # 1~4
            'ca_scale_r': 1.0,      # 0.98~1.02
            'ca_scale_b': 1.0,      # 0.98~1.02
            'gauss_sigma': 0.0,     # 0~20
            'poisson_strength': 0.0,# 0~1
            'motion_len': 0.0,      # 0~31
            'motion_angle': 0.0,    # 0~180
            'rolling_shear': 0.0,   # -0.3~0.3
        }
        self.params = dict(self.default_params)

        # UI
        self.setup_ui()

        # 初始刷新
        self.update_image(force_all=True)

    # -------------------------
    # 基础与 UI
    # -------------------------

    def _invalidate_all_caches(self):
        self.cached_warped = None
        self.cached_warped_sig = None
        self.cached_lens = None
        self.cached_lens_sig = None
        self.cached_grid_pts = None  # for undistortPoints grid (h,w)

    def create_checkerboard_image(self):
        """创建棋盘格图像（备用）"""
        m, n, a, b, c = 12, 16, 28, 18, 60
        width = 2 * c + n * a + (n - 1) * b
        height = 2 * c + m * a + (m - 1) * b
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        for i in range(m):
            for j in range(n):
                x0 = c + j * (a + b)
                y0 = c + i * (a + b)
                image[y0:y0+a, x0:x0+a] = 0
        return image

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # 左侧控制面板
        control = ttk.Notebook(main)
        control.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        self.control_cam = ttk.Frame(control, padding=10)
        self.control_dist = ttk.Frame(control, padding=10)
        self.control_fx = ttk.Frame(control, padding=10)
        control.add(self.control_cam, text="相机/姿态")
        control.add(self.control_dist, text="畸变")
        control.add(self.control_fx, text="缺陷/后效")

        # 右侧图像预览
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # 预览框
        self.image_frame = ttk.LabelFrame(right, text="图像预览", padding=10)
        self.image_frame.grid(row=0, column=0, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)

        # 底部按钮
        btns = ttk.Frame(right)
        btns.grid(row=1, column=0, pady=8, sticky="w")

        ttk.Button(btns, text="加载图像", command=self.load_image).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="保存图像", command=self.save_image).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="保存预设", command=self.save_preset).grid(row=0, column=2, padx=4)
        ttk.Button(btns, text="加载预设", command=self.load_preset).grid(row=0, column=3, padx=4)
        ttk.Button(btns, text="重置参数", command=self.reset_parameters).grid(row=0, column=4, padx=4)

        # 绑定尺寸变化用于自适应预览
        self.image_frame.bind("<Configure>", self._on_preview_resized)

        # ---- 相机/姿态面板 ----
        r = 0
        self._slider(self.control_cam, "FOV(°)", 'fov_deg', 30, 120, 60, r, fmt="{:.1f}")
        r += 1
        self._slider(self.control_cam, "X平移(px)", 'translation_x', -400, 400, 0, r, fmt="{:.1f}")
        r += 1
        self._slider(self.control_cam, "Y平移(px)", 'translation_y', -400, 400, 0, r, fmt="{:.1f}")
        r += 1
        self._slider(self.control_cam, "偏航(yaw,°)", 'rotation_yaw', -60, 60, 0, r, fmt="{:.2f}")
        r += 1
        self._slider(self.control_cam, "俯仰(pitch,°)", 'rotation_pitch', -60, 60, 0, r, fmt="{:.2f}")
        r += 1
        self._slider(self.control_cam, "滚转(roll,°)", 'rotation_roll', -60, 60, 0, r, fmt="{:.2f}")
        r += 1

        # ---- 畸变面板 ----
        r = 0
        self._slider(self.control_dist, "径向 k1", 'k1', -0.6, 0.6, 0, r, fmt="{:.4f}")
        r += 1
        self._slider(self.control_dist, "径向 k2", 'k2', -0.3, 0.3, 0, r, fmt="{:.4f}")
        r += 1
        self._slider(self.control_dist, "径向 k3", 'k3', -0.1, 0.1, 0, r, fmt="{:.5f}")
        r += 1
        self._slider(self.control_dist, "切向 p1", 'p1', -0.01, 0.01, 0, r, fmt="{:.5f}")
        r += 1
        self._slider(self.control_dist, "切向 p2", 'p2', -0.01, 0.01, 0, r, fmt="{:.5f}")
        r += 1

        # ---- 缺陷/后效面板 ----
        r = 0
        self._slider(self.control_fx, "暗角 强度", 'vign_strength', 0.0, 1.0, 0.0, r, fmt="{:.2f}")
        r += 1
        self._slider(self.control_fx, "暗角 指数", 'vign_exponent', 1.0, 4.0, 2.0, r, fmt="{:.2f}")
        r += 1
        self._slider(self.control_fx, "色差 R缩放", 'ca_scale_r', 0.98, 1.02, 1.0, r, fmt="{:.4f}")
        r += 1
        self._slider(self.control_fx, "色差 B缩放", 'ca_scale_b', 0.98, 1.02, 1.0, r, fmt="{:.4f}")
        r += 1
        self._slider(self.control_fx, "高斯噪声 σ", 'gauss_sigma', 0.0, 20.0, 0.0, r, fmt="{:.1f}")
        r += 1
        self._slider(self.control_fx, "泊松噪声 强度", 'poisson_strength', 0.0, 1.0, 0.0, r, fmt="{:.2f}")
        r += 1
        self._slider(self.control_fx, "运动模糊 长度", 'motion_len', 0.0, 31.0, 0.0, r, fmt="{:.1f}")
        r += 1
        self._slider(self.control_fx, "运动模糊 角度", 'motion_angle', 0.0, 180.0, 0.0, r, fmt="{:.1f}")
        r += 1
        self._slider(self.control_fx, "滚动快门 剪切", 'rolling_shear', -0.3, 0.3, 0.0, r, fmt="{:.3f}")
        r += 1

    def _slider(self, parent, label, key, vmin, vmax, vinit, row, fmt="{:.3f}"):
        """创建带标签的 ttk.Scale（含显示数值与防抖更新）"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=4)
        parent.columnconfigure(0, weight=1)

        lbl = ttk.Label(frame, text=f"{label}: {fmt.format(vinit)}")
        lbl.grid(row=0, column=0, sticky="w")

        scale = ttk.Scale(
            frame, from_=vmin, to=vmax, orient=tk.HORIZONTAL,
            command=lambda v, k=key, l=lbl, lab=label, f=fmt: self.on_slider_change(v, k, l, lab, f)
        )
        scale.set(vinit)
        scale.grid(row=0, column=1, sticky="ew", padx=8)
        frame.columnconfigure(1, weight=1)

        if not hasattr(self, 'sliders'):
            self.sliders = {}
        self.sliders[key] = (scale, lbl, fmt, label)

    def on_slider_change(self, value, key, label_widget, label_text, fmt):
        self.params[key] = float(value)
        label_widget.config(text=f"{label_text}: {fmt.format(float(value))}")
        # 防抖：16~30ms 触发一次
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(30, self.update_image)

    def reset_parameters(self):
        for k, (scale, lbl, fmt, label_text) in self.sliders.items():
            v = self.default_params[k]
            scale.set(v)
            lbl.config(text=f"{label_text}: {fmt.format(v)}")
            self.params[k] = v
        self.update_image(force_all=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            initialdir='.',
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All files", "*.*")]
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("错误", "无法加载图像文件")
                return
            self.original_image = img
            self._invalidate_all_caches()
            self.update_image(force_all=True)

    def save_image(self):
        if self.preview_image is None:
            messagebox.showinfo("提示", "没有可保存的图像。")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialdir='.',
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if file_path:
            # 保存原始分辨率处理结果，而不是缩放预览
            out = self.process_full_resolution()
            cv2.imwrite(file_path, out)
            messagebox.showinfo("成功", f"图像已保存到:\n{file_path}")

    def save_preset(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialdir='.',
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not file_path:
            return
        data = {
            "params": self.params,
            "image_info": {
                "width": int(self.original_image.shape[1]),
                "height": int(self.original_image.shape[0])
            }
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("成功", f"预设已保存:\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{e}")

    def load_preset(self):
        file_path = filedialog.askopenfilename(
            initialdir='.',
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            params = data.get("params", {})
            for k, (scale, lbl, fmt, label_text) in self.sliders.items():
                if k in params:
                    v = float(params[k])
                    scale.set(v)
                    lbl.config(text=f"{label_text}: {fmt.format(v)}")
                    self.params[k] = v
            self.update_image(force_all=True)
        except Exception as e:
            messagebox.showerror("错误", f"加载失败：{e}")

    def _on_preview_resized(self, event):
        # 预览区域尺寸变化时重绘（使用现有处理结果缩放）
        self._draw_preview()

    # -------------------------
    # 数学核心 (A)(B)(C) + 后效
    # -------------------------

    # (A) 相机内参
    def make_camera_matrix(self, width, height, fov_deg=60.0):
        # 水平 FOV，等像素
        f = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2.0)
        fx = fy = float(f)
        cx, cy = width / 2.0, height / 2.0
        K = np.array([[fx, 0, cx],
                      [0,  fy, cy],
                      [0,  0,  1]], dtype=np.float32)
        return K

    # (A) 旋转生成单应性
    def build_homography_from_euler(self, K, yaw_deg, pitch_deg, roll_deg):
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch),  np.cos(pitch)]], dtype=np.float32)

        Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                       [           0, 1,          0],
                       [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)

        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll),  np.cos(roll), 0],
                       [0, 0, 1]], dtype=np.float32)

        # 组合顺序：Rx -> Ry -> Rz
        R = Rz @ Ry @ Rx
        Kinv = np.linalg.inv(K)
        Hrot = (K @ R @ Kinv).astype(np.float32)
        return Hrot

    # (B) 正向畸变：undistortPoints 逆映射 + remap
    def apply_lens_distortion(self, image, K, dist):
        # CPU 实现
        if np.allclose(dist, 0, atol=1e-12):
            return image

        h, w = image.shape[:2]
        # 缓存像素网格（同尺寸时复用）
        if self.cached_grid_pts is None or self.cached_grid_pts.shape[:2] != (h, w):
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)  # (N,1,2)
            self.cached_grid_pts = pts

        undist = cv2.undistortPoints(self.cached_grid_pts, K, dist, R=None, P=K)
        undist = undist.reshape(h, w, 2).astype(np.float32)

        map_x = undist[..., 0]
        map_y = undist[..., 1]

        out = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return out

    def apply_lens_distortion_gpu(self, image, K, dist):
        # GPU 实现（undistortPoints 仍在 CPU 上计算映射，再交由 CUDA remap）
        if np.allclose(dist, 0, atol=1e-12):
            return image

        h, w = image.shape[:2]
        if self.cached_grid_pts is None or self.cached_grid_pts.shape[:2] != (h, w):
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
            self.cached_grid_pts = pts

        undist = cv2.undistortPoints(self.cached_grid_pts, K, dist, R=None, P=K)
        undist = undist.reshape(h, w, 2).astype(np.float32)
        map_x = undist[..., 0]
        map_y = undist[..., 1]

        # 上传到 GPU
        g_src = self.gpu.upload(image)
        if g_src is None:
            return self.apply_lens_distortion(image, K, dist)

        g_mapx = self.gpu.upload(map_x)
        g_mapy = self.gpu.upload(map_y)
        if g_mapx is None or g_mapy is None:
            return self.apply_lens_distortion(image, K, dist)

        g_dst = self.gpu.remap(g_src, g_mapx, g_mapy,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
        if g_dst is None:
            return self.apply_lens_distortion(image, K, dist)

        out = self.gpu.download(g_dst)
        return out if out is not None else self.apply_lens_distortion(image, K, dist)

    # (C) 主处理流程（含缓存 + GPU 加速）
    def update_image(self, force_all=False):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
            self.update_job = None

        try:
            h, w = self.original_image.shape[:2]
            K = self.make_camera_matrix(w, h, self.params['fov_deg'])

            # --- 签名 ---
            geom_sig = (
                round(self.params['translation_x'], 6),
                round(self.params['translation_y'], 6),
                round(self.params['rotation_yaw'], 6),
                round(self.params['rotation_pitch'], 6),
                round(self.params['rotation_roll'], 6),
                round(self.params['fov_deg'], 6),
                h, w
            )
            dist_sig = (
                round(self.params['k1'], 8),
                round(self.params['k2'], 8),
                round(self.params['k3'], 8),
                round(self.params['p1'], 8),
                round(self.params['p2'], 8),
                h, w
            )

            # --- 1) 几何：旋转 + 平移（白底） ---
            if force_all or self.cached_warped is None or self.cached_warped_sig != geom_sig:
                Hrot = self.build_homography_from_euler(
                    K,
                    self.params['rotation_yaw'],
                    self.params['rotation_pitch'],
                    self.params['rotation_roll']
                )
                tx = float(self.params['translation_x'])
                ty = float(self.params['translation_y'])
                T = np.array([[1, 0, tx],
                              [0, 1, ty],
                              [0, 0,  1]], dtype=np.float32)
                H = T @ Hrot

                if self.gpu.available:
                    g_src = self.gpu.upload(self.original_image)
                    if g_src is not None:
                        g_warp = self.gpu.warp_perspective(
                            g_src, H, (w, h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255)
                        )
                        if g_warp is not None:
                            warped = self.gpu.download(g_warp)
                        else:
                            warped = cv2.warpPerspective(
                                self.original_image, H, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255)
                            )
                    else:
                        warped = cv2.warpPerspective(
                            self.original_image, H, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255)
                        )
                else:
                    warped = cv2.warpPerspective(
                        self.original_image, H, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255)
                    )

                self.cached_warped = warped
                self.cached_warped_sig = geom_sig
                # 畸变缓存失效
                self.cached_lens = None
                self.cached_lens_sig = None
                self.cached_grid_pts = None

            warped = self.cached_warped

            # --- 2) 光学畸变（正向加畸变） ---
            K = self.make_camera_matrix(w, h, self.params['fov_deg'])
            dist = np.array([
                float(self.params['k1']),
                float(self.params['k2']),
                float(self.params['p1']),
                float(self.params['p2']),
                float(self.params['k3'])
            ], dtype=np.float32)

            if force_all or self.cached_lens is None or self.cached_lens_sig != (dist_sig, self.cached_warped_sig):
                if self.gpu.available:
                    lens_img = self.apply_lens_distortion_gpu(warped, K, dist)
                else:
                    lens_img = self.apply_lens_distortion(warped, K, dist)

                self.cached_lens = lens_img
                self.cached_lens_sig = (dist_sig, self.cached_warped_sig)

            img = self.cached_lens.copy()

            # --- 3) 缺陷/后效 ---
            # 色差（CPU 实现较简洁；GPU 路径收益较小，这里仍用 CPU）
            img = self.apply_chromatic_aberration(img,
                                                  self.params['ca_scale_r'],
                                                  self.params['ca_scale_b'])
            # 暗角（CPU 快速）
            img = self.apply_vignetting(img,
                                        self.params['vign_strength'],
                                        self.params['vign_exponent'])
            # 滚动快门剪切（优先 GPU）
            img = self.apply_rolling_shutter_shear(img, self.params['rolling_shear'])
            # 运动模糊（优先 GPU）
            img = self.apply_motion_blur(img, int(round(self.params['motion_len'])),
                                         self.params['motion_angle'])
            # 噪声（CPU）
            img = self.apply_noise(img, self.params['gauss_sigma'], self.params['poisson_strength'])

            self.final_image = img
            self._draw_preview()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", f"处理过程中发生异常：\n{e}")

    def _draw_preview(self):
        """把 self.final_image 按预览框尺寸缩放后显示"""
        if not hasattr(self, 'final_image'):
            return
        h, w = self.final_image.shape[:2]

        # 预览区域（去掉 LabelFrame 内边距）
        fw = max(self.image_frame.winfo_width() - 20, 1)
        fh = max(self.image_frame.winfo_height() - 40, 1)  # 预留标题高度
        if fw <= 1 or fh <= 1:
            fw, fh = w, h

        scale = min(fw / w, fh / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        disp = cv2.resize(self.final_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.preview_image = disp

        image_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.photo_image)

    def process_full_resolution(self):
        """返回当前参数下的原始分辨率处理结果（避免保存预览的缩放图）"""
        return self.final_image.copy()

    # -------------------------
    # 缺陷效果（含 GPU 尝试 + CPU 回退）
    # -------------------------

    def apply_vignetting(self, image, strength=0.0, exponent=2.0):
        if strength <= 1e-6:
            return image
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        rs = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        rmax = np.sqrt(cx**2 + cy**2)
        mask = 1.0 - strength * (rs / rmax) ** exponent
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

        out = (image.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)
        return out

    def apply_chromatic_aberration(self, image, scale_r=1.0, scale_b=1.0):
        if abs(scale_r - 1.0) < 1e-6 and abs(scale_b - 1.0) < 1e-6:
            return image
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))

        def remap_scale(channel, scale):
            if abs(scale - 1.0) < 1e-6:
                return channel
            map_x = (xs - cx) * scale + cx
            map_y = (ys - cy) * scale + cy
            return cv2.remap(channel, map_x, map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)

        b, g, r = cv2.split(image)
        r2 = remap_scale(r, float(scale_r))
        b2 = remap_scale(b, float(scale_b))
        merged = cv2.merge([b2, g, r2])
        return merged

    def apply_noise(self, image, gauss_sigma=0.0, poisson_strength=0.0):
        out = image.astype(np.float32)

        if gauss_sigma > 1e-6:
            noise = np.random.normal(0.0, gauss_sigma, out.shape).astype(np.float32)
            out = out + noise

        if poisson_strength > 1e-6:
            # 简化的泊松噪声模型：把像素当作强度比例，映射到期望计数 L
            L = 50.0 * float(poisson_strength)  # L 越大，噪声越明显
            if L > 1e-6:
                norm = np.clip(out / 255.0, 0.0, 1.0)
                counts = np.random.poisson(norm * L).astype(np.float32)
                norm_noisy = counts / L
                out = norm_noisy * 255.0

        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_motion_blur(self, image, length=0, angle_deg=0.0):
        L = int(length)
        if L <= 1:
            return image
        # 核大小取奇数
        k = L if L % 2 == 1 else L + 1
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        # 旋转核
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel = kernel / np.sum(kernel + 1e-8)

        s = float(np.sum(kernel))
        if s > 1e-8:
            kernel /= s

        # GPU 路径
        if self.gpu.available:
            g_src = self.gpu.upload(image)
            if g_src is not None:
                g_dst = self.gpu.linear_filter(g_src, kernel)
                if g_dst is not None:
                    out = self.gpu.download(g_dst)
                    if out is not None:
                        return out
        # CPU 回退
        blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        return blurred

    def apply_rolling_shutter_shear(self, image, shear=0.0):
        if abs(shear) < 1e-6:
            return image
        h, w = image.shape[:2]
        # 近似的剪切（x' = x + s*y）
        M = np.array([[1.0, float(shear), 0.0],
                      [0.0, 1.0,          0.0]], dtype=np.float32)

        if self.gpu.available:
            g_src = self.gpu.upload(image)
            if g_src is not None:
                g_dst = self.gpu.warp_affine(
                    g_src, M, (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )
                if g_dst is not None:
                    out = self.gpu.download(g_dst)
                    if out is not None:
                        return out

        out = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return out

    # -------------------------
    # 主程序/入口
    # -------------------------

def main():
    root = tk.Tk()
    app = CameraSimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
