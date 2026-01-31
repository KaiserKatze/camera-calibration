import numpy as np
import cv2
import urllib.request
import os
import shutil


# ==========================================
# 1. 准备工作：下载数据
# ==========================================

# @see: https://github.com/opencv/opencv/blob/master/samples/data/left01.jpg
# @see: https://github.com/opencv/opencv/blob/master/samples/data/left_intrinsics.yml

base_url_raw = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/samples/data/"
file_names = [f'left{i:02d}.jpg' for i in range(1, 15) if i != 10]
yml_name = 'left_intrinsics.yml'

print("--- 步骤 1: 检查并下载数据 ---")

def download_file(filename, timeout_in_seconds=10):
    filepath = os.path.join(os.getcwd(), filename)

    if not os.path.exists(filepath):
        url = base_url_raw + filename
        print(f"正在下载: {filename} ...")
        try:
            with urllib.request.urlopen(url, timeout=timeout_in_seconds) as response, \
                open(filepath, 'wb') as out_file:
                # shutil.copyfileobj 会高效地将流从网络复制到文件
                shutil.copyfileobj(response, out_file)
                out_file.flush()
        except Exception as e:
            print(f"下载 {filename} 失败: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    else:
        print(f'文件 {filename} 已存在，跳过 ...')
    return True

err_list = []
for fname in file_names:
    if not download_file(fname):
        err_list.append(fname)

file_names = list(set(file_names) - set(err_list))
download_file(yml_name)
print("数据下载完毕。\n")

# ==========================================
# 2. 提取角点 & 保存数据到 TXT (数据采集阶段)
# ==========================================
print("--- 步骤 2: 提取角点并保存到 TXT 文件 ---")

# OpenCV 官方 left 系列图片的内角点规格是 9x6
CHECKERBOARD = (9, 6)

# 设置寻找亚像素角点的参数，采用停止准则：最大循环次数30 或 误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 生成模型点 (世界坐标)
# 格式: (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# [新增任务] 保存模型点到 model.txt
# 保存格式：每一行代表一个点的 x y z
np.savetxt("model.txt", objp, fmt='%.6f')
print(f"已保存模型点数据到: model.txt")

image_size = None
valid_files_list = [] # 用于记录哪些图片成功提取了角点，方便后续读取

for fname in file_names:
    print(f'正在处理: {fname} ... ', end='')
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    # 寻找角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        # 亚像素优化
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 将像素坐标写入对应的 txt 文件
        # corners2 的形状是 (N, 1, 2)，保存前 reshape 为 (N, 2)
        txt_name = fname.replace('.jpg', '.txt')
        np.savetxt(txt_name, corners2.reshape(-1, 2), fmt='%.6f')

        valid_files_list.append(txt_name) # 记录生成的txt文件名
        print(f"[{fname}] 提取成功 -> 已保存至 {txt_name}")
    else:
        print(f"[{fname}] 未找到角点 (忽略)")

print(f"数据采集完成，共生成 {len(valid_files_list)} 个角点文件。\n")

# ==========================================
# 3. 从 TXT 读取数据 & 执行相机标定 (计算阶段)
# ==========================================
print("--- 步骤 3: 从 TXT 读取数据并标定 ---")

# 清空内存中的数据，确保我们使用的是从 txt 读取的数据
objpoints_read = []
imgpoints_read = []

# 3.1 读取模型点
if os.path.exists("model.txt"):
    # 读取后是 float64，OpenCV 习惯用 float32
    raw_obj = np.loadtxt("model.txt", dtype=np.float32)
    print(f"已加载 model.txt, 形状: {raw_obj.shape}")
else:
    print("错误：找不到 model.txt")
    exit()

# 3.2 读取像素点
for txt_file in valid_files_list:
    if os.path.exists(txt_file):
        raw_img = np.loadtxt(txt_file, dtype=np.float32)

        # 关键：OpenCV calibrateCamera 要求 imagePoints 的形状是 (N, 1, 2)
        # loadtxt 读出来是 (N, 2)，所以需要 reshape
        raw_img_reshaped = raw_img.reshape(-1, 1, 2)

        objpoints_read.append(raw_obj)          # 每张图对应的都是同一个模型板子
        imgpoints_read.append(raw_img_reshaped) # 每张图特定的像素角点
    else:
        print(f"警告: 找不到文件 {txt_file}")

# 3.3 执行标定
ret, mtx_calc, dist_calc, rvecs, tvecs = cv2.calibrateCamera(
    objpoints_read, imgpoints_read, image_size, None, None
)

print(f"标定重投影误差 (RMS): {ret:.4f} 像素")
print("计算得到的内参矩阵 (K):\n", mtx_calc)
print("计算得到的畸变系数 (D):\n", dist_calc.ravel())
print("\n")

# ==========================================
# 4. 读取官方真值 (left_intrinsics.yml) 并对比
# ==========================================
print("--- 步骤 4: 读取官方真值并对比 ---")

if not os.path.exists(yml_name):
    print("错误：找不到真值文件，无法对比。")
else:
    fs = cv2.FileStorage(yml_name, cv2.FILE_STORAGE_READ)

    # 从 YML 读取矩阵和畸变系数
    mtx_true = fs.getNode("camera_matrix").mat()
    dist_true = fs.getNode("distortion_coefficients").mat()
    fs.release()

    if mtx_true is None or dist_true is None:
        print("读取 YML 失败，文件格式可能不兼容。")
    else:
        # ==========================================
        # 5. 计算误差分析
        # ==========================================
        def analyze_error(name, val_calc, val_true):
            abs_err = abs(val_calc - val_true)
            # 避免除以0
            rel_err = (abs_err / abs(val_true)) * 100 if val_true != 0 else 0.0
            print(f"{name:<15} | 计算值: {val_calc:10.4f} | 真值: {val_true:10.4f} | 绝对误差: {abs_err:8.4f} | 相对误差: {rel_err:6.2f}%")

        print(f"{'参数项':<15} | {'计算结果':<15} | {'官方真值':<15} | {'绝对误差':<8} | {'相对误差':<8}")
        print("-" * 100)

        # 对比焦距 fx, fy
        analyze_error("Fx (焦距X)", mtx_calc[0, 0], mtx_true[0, 0])
        analyze_error("Fy (焦距Y)", mtx_calc[1, 1], mtx_true[1, 1])

        # 对比光心 cx, cy
        analyze_error("Cx (光心X)", mtx_calc[0, 2], mtx_true[0, 2])
        analyze_error("Cy (光心Y)", mtx_calc[1, 2], mtx_true[1, 2])

        print("-" * 100)

        # 对比畸变系数 (k1, k2, p1, p2, k3)
        # 注意：计算结果可能包含 k3 也可能不包含（取决于标志位），通常前5个最重要
        d_calc = dist_calc.ravel()
        d_true = dist_true.ravel()
        labels = ["k1", "k2", "p1", "p2", "k3"]
        for i in range(min(len(d_calc), len(d_true), 5)):
            analyze_error(f"Dist {labels[i]}", d_calc[i], d_true[i])

print("\n实验结束。")
