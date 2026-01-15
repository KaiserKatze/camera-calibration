import numpy as np

def normalize_points(points):
    """
    实现论文 Section 6.1 描述的各向同性归一化 (Isotropic Scaling)。

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

def compute_fundamental_matrix_normalized(p1, p2):
    """
    实现论文 Section 2 和 Section 3 的 Normalized 8-point Algorithm。

    Args:
        p1: 第一幅图像中的匹配点，形状 (N, 2)。
        p2: 第二幅图像中的匹配点，形状 (N, 2)。

    Returns:
        F: 3x3 基础矩阵 (Fundamental Matrix)。
    """
    if p1.shape[0] < 8 or p2.shape[0] < 8:
        raise ValueError("At least 8 point matches are required.")

    # --- Step 1: 归一化输入坐标 [cite: 90, 160] ---
    p1_norm, T1 = normalize_points(p1)
    p2_norm, T2 = normalize_points(p2)

    # --- Step 2: 线性求解 (Linear Solution) [cite: 73] ---
    # 构建方程矩阵 A，求解 Af = 0
    # 对应方程 (1): u'^T F u = 0，这里 p2 是 u'，p1 是 u
    # A 的每一行对应一对点: [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] [cite: 46]
    # 注意：引用 [46] 的顺序与标准实现可能略有不同，这里使用标准的 Kronecker product 顺序
    # 即 (x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1)

    N = p1_norm.shape[0]
    A = np.zeros((N, 9))

    x1, y1 = p1_norm[:, 0], p1_norm[:, 1]
    x2, y2 = p2_norm[:, 0], p2_norm[:, 1]

    A[:, 0] = x2 * x1
    A[:, 1] = x2 * y1
    A[:, 2] = x2
    A[:, 3] = y2 * x1
    A[:, 4] = y2 * y1
    A[:, 5] = y2
    A[:, 6] = x1
    A[:, 7] = y1
    A[:, 8] = 1

    # 使用 SVD 求解最小二乘解 [cite: 61, 65]
    # f 是 A^T A 最小特征值对应的特征向量
    U, S, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # --- Step 3: 强制秩约束 (Constraint Enforcement) [cite: 66, 69] ---
    # 基础矩阵必须是秩为 2 的奇异矩阵。
    # 对 F_norm 进行 SVD: F = U * D * V^T
    U_f, S_f, Vt_f = np.linalg.svd(F_norm)

    # 将最小的奇异值置为 0 [cite: 71]
    S_f[2] = 0

    # 重构 F_norm
    F_norm = U_f @ np.diag(S_f) @ Vt_f

    # --- Step 4: 去归一化 (Denormalization)  ---
    # 论文公式: F = T'^T * F_norm * T
    # 这里 T' 是 T2，T 是 T1
    F = T2.T @ F_norm @ T1

    # 通常将 F 的最后一个元素归一化为 1 (可选)
    if F[2, 2] != 0:
        F = F / F[2, 2]

    return F

# --- 示例用法 ---
if __name__ == "__main__":
    # 生成一些模拟数据进行测试
    # 真实 3D 点
    points_3d = np.random.rand(10, 3)

    # 简单的相机矩阵 (仅用于生成测试点)
    P1 = np.array([[1000, 0, 256, 0], [0, 1000, 256, 0], [0, 0, 1, 0]])
    P2 = np.array([[1000, 0, 256, 100], [0, 1000, 256, 0], [0, 0, 1, 0]])

    def project(P, X):
        x = P @ np.hstack((X, np.ones((X.shape[0], 1)))).T
        x = x[:2] / x[2]
        return x.T

    pts1 = project(P1, points_3d)
    pts2 = project(P2, points_3d)

    # 添加少量噪声
    pts1 += np.random.normal(0, 0.5, pts1.shape)
    pts2 += np.random.normal(0, 0.5, pts2.shape)

    # 计算基础矩阵
    F_matrix = compute_fundamental_matrix_normalized(pts1, pts2)

    print("Computed Fundamental Matrix F:\n", F_matrix)

    # 验证 x2^T * F * x1 ≈ 0
    p1_homo = np.hstack((pts1[0], 1))
    p2_homo = np.hstack((pts2[0], 1))
    error = p2_homo.T @ F_matrix @ p1_homo
    print(f"\nEpipolar constraint error for point 0: {error:.6f}")
