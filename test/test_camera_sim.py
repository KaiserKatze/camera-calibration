#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk
import sys

class CameraSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("相机透视与畸变模拟器")
        self.root.geometry("1200x700")

        # 加载默认图像（使用您提供的棋盘格图像）
        self.original_image = cv2.imread('square_grid.png')
        if self.original_image is None:
            # 如果找不到文件，创建一个示例图像
            self.original_image = self.create_checkerboard_image()

        self.processed_image = self.original_image.copy()
        self.setup_ui()
        self.update_image()

    def create_checkerboard_image(self):
        """创建棋盘格图像（备用方法）"""
        # 参数与您提供的代码一致
        m, n, a, b, c = 8, 8, 25, 30, 50
        width = 2 * c + n * a + (n - 1) * b
        height = 2 * c + m * a + (m - 1) * b

        # 创建白色背景
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 绘制黑色方格
        for i in range(m):
            for j in range(n):
                x_start = c + j * (a + b)
                y_start = c + i * (a + b)
                x_end = x_start + a
                y_end = y_start + a

                image[y_start:y_end, x_start:x_end] = 0  # 黑色方格

        return image

    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 控制面板框架
        control_frame = ttk.LabelFrame(main_frame, text="相机参数控制", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.N, tk.S), padx=(0, 10))

        # 图像显示框架
        image_frame = ttk.LabelFrame(main_frame, text="图像预览", padding="10")
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        # 相机参数（初始值）
        self.params = {
            'translation_x': 0.0,    # X平移
            'translation_y': 0.0,    # Y平移
            'rotation_yaw': 0.0,     # 偏航角
            'rotation_pitch': 0.0,   # 俯仰角
            'rotation_roll': 0.0,    # 滚转角
            'k1': 0.0,              # 径向畸变系数1
            'k2': 0.0,              # 径向畸变系数2
            'k3': 0.0,              # 径向畸变系数3
            'p1': 0.0,              # 切向畸变系数1
            'p2': 0.0               # 切向畸变系数2
        }

        # 创建滑块控件
        self.sliders = {}
        row = 0

        # 平移参数
        self.create_slider(control_frame, "X平移", "translation_x", -30, 30, 0, row)
        row += 1
        self.create_slider(control_frame, "Y平移", "translation_y", -30, 30, 0, row)
        row += 1

        # 旋转参数
        self.create_slider(control_frame, "偏航角", "rotation_yaw", -45, 45, 0, row)
        row += 1
        self.create_slider(control_frame, "俯仰角", "rotation_pitch", -45, 45, 0, row)
        row += 1
        self.create_slider(control_frame, "滚转角", "rotation_roll", -45, 45, 0, row)
        row += 1

        # 径向畸变参数
        self.create_slider(control_frame, "径向畸变 k1", "k1", -5, 5, 0, row)
        row += 1
        self.create_slider(control_frame, "径向畸变 k2", "k2", -5, 5, 0, row)
        row += 1
        self.create_slider(control_frame, "径向畸变 k3", "k3", -5, 5, 0, row)
        row += 1

        # 切向畸变参数
        self.create_slider(control_frame, "切向畸变 p1", "p1", -5, 5, 0, row)
        row += 1
        self.create_slider(control_frame, "切向畸变 p2", "p2", -5, 5, 0, row)
        row += 1

        # 按钮框架
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        # 重置按钮
        reset_btn = ttk.Button(button_frame, text="重置参数", command=self.reset_parameters)
        reset_btn.grid(row=0, column=0, padx=5)

        # 保存按钮
        save_btn = ttk.Button(button_frame, text="保存图像", command=self.save_image)
        save_btn.grid(row=0, column=1, padx=5)

        # 加载图像按钮
        load_btn = ttk.Button(button_frame, text="加载图像", command=self.load_image)
        load_btn.grid(row=0, column=2, padx=5)

        # 图像显示标签
        self.image_label = ttk.Label(image_frame)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_slider(self, parent, label_text, param_name, from_, to_, initial, row):
        """创建带标签的滑块控件"""
        # 参数标签
        label = ttk.Label(parent, text=f"{label_text}: {self.params[param_name]:.4f}")
        label.grid(row=row, column=0, sticky=tk.W, pady=2)

        # 滑块控件
        slider = ttk.Scale(parent, from_=from_, to=to_, orient=tk.HORIZONTAL,
                           command=lambda v, p=param_name, l=label, t=label_text:
                           self.on_slider_change(v, p, l, t))
        slider.set(initial)
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

        # 存储引用
        self.sliders[param_name] = (slider, label)

        # 配置列权重
        parent.columnconfigure(1, weight=1)

    def on_slider_change(self, value, param_name, label, label_text):
        """滑块值改变时的回调函数"""
        self.params[param_name] = float(value)
        label.config(text=f"{label_text}: {float(value):.4f}")
        self.update_image()

    def reset_parameters(self):
        """重置所有参数到初始值"""
        for param_name, (slider, label) in self.sliders.items():
            slider.set(0.0)
            # 更新标签文本需要知道原始标签文本，这里简化处理
            if param_name.startswith('translation'):
                label_text = "X平移" if "x" in param_name else "Y平移"
            elif param_name.startswith('rotation'):
                label_text = "偏航角" if "yaw" in param_name else "俯仰角" if "pitch" in param_name else "滚转角"
            elif param_name.startswith('k'):
                label_text = f"径向畸变 {param_name}"
            elif param_name.startswith('p'):
                label_text = f"切向畸变 {param_name}"
            else:
                label_text = param_name

            label.config(text=f"{label_text}: 0.0000")

        self.params = {key: 0.0 for key in self.params.keys()}
        self.update_image()

    def load_image(self):
        """加载用户选择的图像"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All files", "*.*")]
        )

        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.update_image()
            else:
                print("无法加载图像文件")

    def save_image(self):
        """保存处理后的图像"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )

        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            print(f"图像已保存到: {file_path}")

    def update_image(self):
        """根据当前参数更新图像，确保背景为白色"""
        # 获取图像尺寸
        height, width = self.original_image.shape[:2]

        # 创建一个全白的背景图像
        white_background = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 应用透视变换
        translation_x = self.params['translation_x']
        translation_y = self.params['translation_y']

        # 计算旋转矩阵
        yaw = np.radians(self.params['rotation_yaw'])
        pitch = np.radians(self.params['rotation_pitch'])
        roll = np.radians(self.params['rotation_roll'])

        # 定义原始点和目标点
        src_points = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)

        # 应用平移和旋转
        center = np.array([width/2, height/2])

        # 计算变换后的点
        dst_points = []
        for x, y in src_points:
            # 相对中心点的坐标
            x_rel = x - center[0]
            y_rel = y - center[1]

            # 应用旋转（简化处理）
            x_rot = x_rel * np.cos(yaw) - y_rel * np.sin(pitch)
            y_rot = x_rel * np.sin(roll) + y_rel * np.cos(roll)

            # 应用平移并移回绝对坐标
            x_new = x_rot + center[0] + translation_x
            y_new = y_rot + center[1] + translation_y

            dst_points.append([x_new, y_new])

        dst_points = np.array(dst_points, dtype=np.float32)

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 应用透视变换，设置白色背景
        transformed = cv2.warpPerspective(
            self.original_image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # 白色背景
        )

        # 应用镜头畸变
        self.processed_image = self.apply_lens_distortion(transformed)

        # 转换图像格式用于显示
        image_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo_image = ImageTk.PhotoImage(pil_image)

        # 更新显示
        self.image_label.configure(image=photo_image)
        self.image_label.image = photo_image

    def apply_lens_distortion(self, image):
        """应用镜头畸变效果"""
        if all(v == 0 for k, v in self.params.items() if k.startswith(('k', 'p'))):
            return image

        # 获取图像尺寸
        height, width = image.shape[:2]

        # 生成相机矩阵（假设图像中心为光学中心）
        camera_matrix = np.array([
            [width, 0, width/2],
            [0, height, height/2],
            [0, 0, 1]
        ], dtype=np.float32)

        # 畸变系数
        dist_coeffs = np.array([
            self.params['k1'],
            self.params['k2'],
            self.params['p1'],
            self.params['p2'],
            self.params['k3']
        ], dtype=np.float32)

        # 应用畸变校正
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

        return undistorted

def main():
    """主函数"""
    root = tk.Tk()
    app = CameraSimulatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
