#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from PIL import Image, ImageDraw

# 定义常量参数
m = 8  # 网格行数
n = 8  # 网格列数
a = 25  # 每个黑色正方形的边长（像素）
b = 30  # 相邻正方形之间的间距（像素）
c = 50  # 正方形与图片边缘的最小距离（像素）

# 计算图片总尺寸
image_width = 2 * c + n * a + (n - 1) * b
image_height = 2 * c + m * a + (m - 1) * b

if __name__ == '__main__':
    print(f'Python version: {sys.version}')
    print('sys.path=', sys.path)

    # 创建白色背景图像
    image = Image.new('RGB', (image_width, image_height), 'white')
    draw = ImageDraw.Draw(image)

    # 绘制黑色正方形网格
    for i in range(m):
        for j in range(n):
            # 计算当前正方形的左上角坐标
            x_start = c + j * (a + b)
            y_start = c + i * (a + b)

            # 计算当前正方形的右下角坐标
            x_end = x_start + a
            y_end = y_start + a

            # 绘制黑色正方形
            draw.rectangle([x_start, y_start, x_end, y_end], fill='black')

    # 保存图像为PNG文件
    image.save('square_grid.png')
    print(f'图片已生成并保存为 square_grid.png')
    print(f'图片尺寸: {image_width} * {image_height} 像素')

    # 可选：显示图像（需要在有图形界面的环境中运行）
    image.show()
