#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_blas.h>

#define CONSTANT_PI 3.14159265358979323846f
#define ARG_INPUT
#define ARG_OUTPUT
#define ARG_INPUT_OUTPUT

// 每一个列向量都是一个点的齐次坐标
using Point3dMatrix = Eigen::Matrix3Xf;
// 每一个列向量都是一个点的非齐次坐标
using Point2dMatrix = Eigen::Matrix2Xf;
// 用来构造系数矩阵
namespace Eigen {
    using Vector6f = Eigen::Matrix<float, 6, 1>;
    using RowVector6f = Eigen::Matrix<float, 1, 6>;
    using MatrixX6f = Eigen::Matrix<float, Eigen::Dynamic, 6>;
    using Vector9f = Eigen::Matrix<float, 9, 1>;
    using MatrixX9f = Eigen::Matrix<float, Eigen::Dynamic, 9>;
}

Eigen::Matrix3f MakeIntrinsicMatrix(float alpha, float beta, float gamma, float u0, float v0) {
    Eigen::Matrix3f matInParams;  // 内参矩阵
    matInParams << alpha, gamma, u0,
                   0.0f,  beta,  v0,
                   0.0f,  0.0f,  1.0f;
    return matInParams;
}

struct ExParams {
    float rx;
    float ry;
    float rz;
    float tx;
    float ty;
    float tz;

    /**
     * 使用 AngleAxis 转换旋转向量到旋转矩阵
     */
    Eigen::Matrix3f GetRotation() const {
        Eigen::Vector3f rVec(rx, ry, rz);
        float angle = rVec.norm();
        if (angle < 1e-8f) {
            return Eigen::Matrix3f::Identity();
        }
        return Eigen::AngleAxisf(angle, rVec.normalized()).toRotationMatrix();
    }

    Eigen::Vector3f GetTranslation() const {
        return Eigen::Vector3f(tx, ty, tz);
    }
};

// 罗德里格斯变换变换 (Rodrigues Vector -> Rotation Matrix)
Eigen::Matrix3f Rodrigues(const Eigen::Vector3f& rVec) {
    float angle = rVec.norm();
    if (angle < 1e-8f) {
        return Eigen::Matrix3f::Identity();
    }
    return Eigen::AngleAxisf(angle, rVec.normalized()).toRotationMatrix();
}

// 罗德里格斯变换逆变换 (Rotation Matrix -> Rodrigues Vector)
Eigen::Vector3f InvRodrigues(const Eigen::Matrix3f& R) {
    Eigen::AngleAxisf angleAxis(R);
    return angleAxis.angle() * angleAxis.axis();
}

void Homo2Nonhomo(ARG_INPUT const Eigen::Matrix3Xf& homo, ARG_OUTPUT Eigen::Matrix2Xf& nonhomo) {
    if (homo.rows() != 3) {
        throw std::runtime_error("Homo2Nonhomo: homo must be 3xN.");
    }
    nonhomo.resize(2, homo.cols());
    // 使用行数组类型 (1, Dynamic) 以匹配 homo.row(i) 的维度 (1, N)
    // 不能使用 ArrayXf 作为 denom 的类型，否则会导致列向量 (N, 1)，从而在除法时触发维度断言错误
    Eigen::Array<float, 1, Eigen::Dynamic> denom = homo.row(2).array();
    denom = (denom.abs() < 1e-12).select(1e-12, denom); // 如果绝对值小于阈值，设为 1e-12
    nonhomo.row(0).array() = homo.row(0).array() / denom;
    nonhomo.row(1).array() = homo.row(1).array() / denom;
}

void Nonhomo2Homo(ARG_INPUT const Eigen::Matrix2Xf& nonhomo, ARG_OUTPUT Eigen::Matrix3Xf& homo) {
    if (nonhomo.rows() != 2) {
        throw std::runtime_error("Nonhomo2Homo: nonhomo must be 3xN.");
    }
    homo.resize(3, nonhomo.cols());
    homo.row(0).array() = nonhomo.row(0).array();
    homo.row(1).array() = nonhomo.row(1).array();
    homo.row(2) = Eigen::RowVectorXf::Ones( nonhomo.cols() );
}

/**
 * 进行各向同性归一化，平移点使形心位于原点，缩放使平均距离为 sqrt(2)
 *
 * @param points: 点的齐次坐标（断言 W 坐标为 1）
 * @return 变换矩阵
 */
Eigen::Matrix3f IsotropicScalingNormalize(ARG_INPUT_OUTPUT Eigen::Matrix3Xf& points) {
    if (points.rows() != 3) {
        throw std::runtime_error("IsotropicScalingNormalize: points must be 3xN.");
    }
    static const float tiny = 1e-8;

    // 1. 计算重心
    float centroidX = points.row(0).mean();
    float centroidY = points.row(1).mean();

    // 2. 计算平均距离 (相对于重心)
    Eigen::Matrix2Xf centeredPoints = points.topRows(2);
    centeredPoints.row(0).array() -= centroidX;
    centeredPoints.row(1).array() -= centroidY;
    Eigen::VectorXf distances = centeredPoints.colwise().norm();
    float meanDistance = distances.mean();

    // 3. 计算缩放因子
    float scale = (meanDistance < tiny) ? 1.0f : (std::sqrt(2.0f) / meanDistance);

    // 4. 构造变换矩阵
    Eigen::Matrix3f sMat;
    sMat << scale,  0.0f,   -scale * centroidX,
            0.0f,   scale,  -scale * centroidY,
            0.0f,   0.0f,   1.0f;

    // 5. 应用变换矩阵
    points = sMat * points;

    return sMat;
}

struct DistortFunction {  // 基类
    /**
     * @param points: 理想的（无畸变的）像素点坐标，归一化图像坐标系
     */
    virtual void Distort(ARG_INPUT_OUTPUT Eigen::Matrix3Xf& points) const {}

    virtual void DistortPoint(ARG_INPUT_OUTPUT Eigen::Vector2f& point) const {}
};

/**
 * 使用简化的 Brown-Conrady 畸变模型（只有2个径向畸变系数，没有切向畸变系数）
 */
struct DistortFunctionBrownConrady : DistortFunction {
    float k1, k2;

    DistortFunctionBrownConrady(float _k1, float _k2) : k1(_k1), k2(_k2) {}

    virtual void Distort(ARG_INPUT_OUTPUT Eigen::Matrix3Xf& points) const override {
        if (points.rows() != 3) {
            throw std::runtime_error("DistortFunctionBrownConrady::Distort: points must be 3xN.");
        }
        Eigen::Matrix2Xf nonhomo;
        Homo2Nonhomo(points, nonhomo);
        for (int i = 0; i < nonhomo.cols(); ++i) {
            Eigen::Vector2f point = nonhomo.col(i);
            DistortPoint(point);
            nonhomo.col(i) = point;
        }
        points.row(0) = nonhomo.row(0);
        points.row(1) = nonhomo.row(1);
    }

    virtual void DistortPoint(ARG_INPUT_OUTPUT Eigen::Vector2f& point) const override {
        float r2 = point.squaredNorm();
        float coeff = 1.0f + k1 * r2 + k2 * r2 * r2;
        point *= coeff;
    }
};

/**
 * 将模型点投影到像平面上，得到像素点的非齐次坐标
 *
 * @param modelPointsInWorldCoordinates: 模型点在世界坐标系中的齐次坐标（默认 Z 坐标为 0）
 * @param iMat: 相机内部参数矩阵
 * @param rMat: 旋转矩阵
 * @param tVec: 平移向量
 */
auto Project(const Eigen::Matrix3Xf& modelPointsInWorldCoordinates,
             const Eigen::Matrix3f& iMat, const Eigen::Matrix3f& rMat, const Eigen::Vector3f& tVec,
             const DistortFunction& distortFunction) {
    if (modelPointsInWorldCoordinates.rows() != 3) {
        throw std::runtime_error("Project: modelPointsInWorldCoordinates must be 3xN.");
    }
    Eigen::Matrix3f rtMat;
    Eigen::Matrix3Xf modelPointsInCameraCoordinates;
    Eigen::Matrix3Xf pixelPointsInImageCoordinates;
    Eigen::Matrix2Xf pixelPointsInPixelCoordinates;

    rtMat.col(0) = rMat.col(0);
    rtMat.col(1) = rMat.col(1);
    rtMat.col(2) = tVec;
    // 将模型点的齐次坐标从世界坐标系变换到相机坐标系
    modelPointsInCameraCoordinates = rtMat * modelPointsInWorldCoordinates;
    // 套用畸变模型
    distortFunction.Distort(modelPointsInCameraCoordinates);
    // 将模型点投影到像平面，得到像素点在像平面坐标系上的齐次坐标
    pixelPointsInImageCoordinates = iMat * modelPointsInCameraCoordinates;
    // 归一化得到像素点的非齐次坐标
    Homo2Nonhomo(pixelPointsInImageCoordinates, pixelPointsInPixelCoordinates);
    return pixelPointsInPixelCoordinates;
}

Eigen::Matrix3f InferHomography(ARG_INPUT const Eigen::Matrix3Xf& modelPointsInWorldCoordinates,
                                ARG_INPUT const Eigen::Matrix3Xf& pixelPointsInPixelCoordinates) {
    if (modelPointsInWorldCoordinates.rows() != 3) {
        throw std::runtime_error("InferHomography: modelPointsInWorldCoordinates must be 3xN.");
    }
    if (pixelPointsInPixelCoordinates.rows() != 3) {
        throw std::runtime_error("InferHomography: pixelPointsInPixelCoordinates must be 3xN.");
    }
    if (modelPointsInWorldCoordinates.cols() != pixelPointsInPixelCoordinates.cols()) {
        throw std::runtime_error("InferHomography: number of model points and pixel points must match.");
    }

    // 1. 数据准备与归一化
    Eigen::Matrix3Xf modelHomo = modelPointsInWorldCoordinates;
    Eigen::Matrix3Xf pixelHomo = pixelPointsInPixelCoordinates;
    Eigen::Matrix3f T_model = IsotropicScalingNormalize(modelHomo);
    Eigen::Matrix3f T_pixel = IsotropicScalingNormalize(pixelHomo);

    int nColsModel = modelHomo.cols();
    // 2. 构造矩阵 matL (2N x 9)
    Eigen::MatrixX9f matL(2 * nColsModel, 9);
    matL.setZero();
    for (int i = 0; i < nColsModel; ++i) {
        float x = modelHomo(0, i);
        float y = modelHomo(1, i);
        float w = modelHomo(2, i);
        float u = pixelHomo(0, i);
        float v = pixelHomo(1, i);
        // Row 1: [X, Y, W, 0, 0, 0, -uX, -uY, -uW]
        matL.row(2 * i)     << x, y, w, 0, 0, 0, -u * x, -u * y, -u * w;
        // Row 2: [0, 0, 0, X, Y, W, -vX, -vY, -vW]
        matL.row(2 * i + 1) << 0, 0, 0, x, y, w, -v * x, -v * y, -v * w;
    }

    if (matL.rows() != 2 * nColsModel || matL.cols() != 9) {
        throw std::runtime_error("InferHomography: matL shape mismatch.");
    }

    // 3. SVD 求解，取最小奇异值对应的右奇异向量
    Eigen::JacobiSVD<Eigen::MatrixX9f> svd(matL, Eigen::ComputeThinV);
    Eigen::Vector9f h = svd.matrixV().col(svd.matrixV().cols() - 1);

    if (h.size() != 9) {
        throw std::runtime_error("InferHomography: SVD solution vector size mismatch.");
    }

    // 4. 重构 H_norm 并去归一化
    Eigen::Matrix3f H_norm;
    H_norm << h(0), h(1), h(2),
              h(3), h(4), h(5),
              h(6), h(7), h(8);

    // H = inv(T_pixel) * H_norm * T_model
    Eigen::Matrix3f H = T_pixel.inverse() * H_norm * T_model;

    // 归一化 H，使 H(2,2) = 1 (如果非零)
    if (std::abs(H(2, 2)) > 1e-8) {
        H /= H(2, 2);
    }

    return H;
}

// 定义上下文结构体，用于在 GSL C 回调中传递 C++ 对象
struct CalibrationContext {
    const Eigen::Matrix3Xf* modelPoints;
    const std::vector<Eigen::Matrix3Xf>* pixelPointsList;
    size_t numViews;
};

void ExtractIntrinsicParams(ARG_INPUT const std::vector<Eigen::Matrix3f>& listHomography,
                            ARG_INPUT const Eigen::Matrix3Xf& modelPointsInWorldCoordinates,
                            ARG_INPUT const std::vector<Eigen::Matrix3Xf>& listPixelPointsInPixelCoordinates,
                            ARG_OUTPUT Eigen::Matrix3f& intrinsicMatrix,
                            ARG_OUTPUT Eigen::Vector2f& radialDistortionCoeffcients) {
    if (modelPointsInWorldCoordinates.rows() != 3) {
        throw std::runtime_error("ExtractIntrinsicParams: modelPointsInWorldCoordinates must be 3xN.");
    }
    if (listHomography.size() != listPixelPointsInPixelCoordinates.size()) {
        throw std::runtime_error("ExtractIntrinsicParams: listHomography and listPixelPointsInPixelCoordinates size mismatch.");
    }

    auto numViews = listHomography.size();
    const size_t numPointsPerView = modelPointsInWorldCoordinates.cols();

    if (numViews < 3) {
        std::cerr << "Warning: At least 3 views are required for calibration (technically 2 if skew is 0, but 3 is safer).\n";
    }

    std::vector<Eigen::Vector3f> rVecs;
    std::vector<Eigen::Vector3f> tVecs;

    rVecs.reserve(numViews);
    tVecs.reserve(numViews);

    {  // make_initial_guess
        // v_ij 函数的 lambda 实现，返回 1x6 向量
        const auto create_v_ij = [](const Eigen::Matrix3f& matHomography, int i, int j) -> Eigen::RowVector6f {
            Eigen::RowVector6f v;
            v(0) = matHomography(0, i) * matHomography(0, j);
            v(1) = matHomography(0, i) * matHomography(1, j) + matHomography(1, i) * matHomography(0, j);
            v(2) = matHomography(1, i) * matHomography(1, j);
            v(3) = matHomography(2, i) * matHomography(0, j) + matHomography(0, i) * matHomography(2, j);
            v(4) = matHomography(2, i) * matHomography(1, j) + matHomography(1, i) * matHomography(2, j);
            v(5) = matHomography(2, i) * matHomography(2, j);
            return v;
        };

        // 构造系数矩阵
        Eigen::MatrixX6f matV(2 * numViews, 6);
        for (decltype(numViews) k = 0; k < numViews; ++k) {
            const Eigen::Matrix3f& matHomography = listHomography[k];
            matV.row(2 * k) = create_v_ij(matHomography, 0, 1);
            matV.row(2 * k + 1) = create_v_ij(matHomography, 0, 0) - create_v_ij(matHomography, 1, 1);
        }

        if (matV.rows() != 2 * numViews || matV.cols() != 6) {
             throw std::runtime_error("ExtractIntrinsicParams: matV shape mismatch.");
        }

        Eigen::JacobiSVD<Eigen::MatrixX6f> svd(matV, Eigen::ComputeThinV);
        // b = [B11, B12, B22, B13, B23, B33]
        Eigen::Vector6f b = svd.matrixV().col(svd.matrixV().cols() - 1); // 最后一列

        if (b.size() != 6) {
             throw std::runtime_error("ExtractIntrinsicParams: b vector size mismatch.");
        }

        if (b(0) < 0) {  // 确保 B11 为正
            b = -b;
        }

        {  // 计算基本矩阵、中间变量
            float B11 = b(0);
            float B12 = b(1);
            float B22 = b(2);
            float B13 = b(3);
            float B23 = b(4);
            float B33 = b(5);

            float mid1 = B12 * B13 - B11 * B23;
            float mid2 = B11 * B22 - B12 * B12;
            float v0 = mid1 / mid2;
            float lambda = B33 - (B13 * B13 + v0 * mid1) / B11;
            float alpha2 = lambda / B11;
            float alpha = std::sqrt(alpha2);
            float beta = std::sqrt(lambda * B11 / mid2);
            float gamma = -B12 * alpha2 * beta / lambda;
            float u0 = gamma * v0 / beta - B13 * alpha2 / lambda;

            intrinsicMatrix <<  alpha, gamma, u0,
                                0.0f,  beta,  v0,
                                0.0f,  0.0f,  1.0f;
            radialDistortionCoeffcients << 0.0f, 0.0f;
        }

        Eigen::Matrix3f invIntrinsicMatrix = intrinsicMatrix.inverse();

        const auto approximate_rotation_matrix = [](Eigen::Matrix3f& mat) -> void {
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Eigen::Matrix3f& U = svd.matrixU();
            const Eigen::Matrix3f& V = svd.matrixV();
            mat = U * V.transpose();
        };

        for (const Eigen::Matrix3f& matHomography : listHomography) {
            const Eigen::Vector3f& h1 = matHomography.row(0);
            const Eigen::Vector3f& h2 = matHomography.row(1);
            const Eigen::Vector3f& h3 = matHomography.row(2);
            Eigen::Vector3f invKh1 = invIntrinsicMatrix * h1;
            Eigen::Vector3f invKh2 = invIntrinsicMatrix * h2;
            Eigen::Vector3f invKh3 = invIntrinsicMatrix * h3;
            float lambda1 = 1.0f / invKh1.norm();
            Eigen::Vector3f r1 = lambda1 * invKh1;
            Eigen::Vector3f r2 = lambda1 * invKh2;
            Eigen::Vector3f r3 = r1.cross(r2);
            Eigen::Vector3f t = lambda1 * invKh3;
            Eigen::Matrix3f matR;
            matR.col(0) = r1;
            matR.col(1) = r2;
            matR.col(2) = r3;
            approximate_rotation_matrix(matR);
            rVecs.push_back(InvRodrigues(matR));
            tVecs.push_back(t);
        }
    }

    // 参数数量
    const size_t p = 7 + 6 * numViews;
    // 残差数量
    const size_t n = 2 * numPointsPerView * numViews;

    gsl_vector* x;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params;
    const gsl_multifit_nlinear_type* T;
    gsl_multifit_nlinear_workspace* w;

    // 初始化参数向量 x
    x = gsl_vector_alloc(p);
    gsl_vector_set(x, 0, intrinsicMatrix(0, 0)); // alpha
    gsl_vector_set(x, 1, intrinsicMatrix(0, 1)); // gamma
    gsl_vector_set(x, 2, intrinsicMatrix(0, 2)); // u0
    gsl_vector_set(x, 3, intrinsicMatrix(1, 1)); // beta
    gsl_vector_set(x, 4, intrinsicMatrix(1, 2)); // v0
    gsl_vector_set(x, 5, 0.0f);  // 初始畸变系数设为 0
    gsl_vector_set(x, 6, 0.0f);
    for (size_t i = 0; i < numViews; ++i) {
        size_t base = 7 + i * 6;
        gsl_vector_set(x, base + 0, rVecs[i].x());
        gsl_vector_set(x, base + 1, rVecs[i].y());
        gsl_vector_set(x, base + 2, rVecs[i].z());
        gsl_vector_set(x, base + 3, tVecs[i].x());
        gsl_vector_set(x, base + 4, tVecs[i].y());
        gsl_vector_set(x, base + 5, tVecs[i].z());
    }

    // 配置 GSL 求解器
    // 准备上下文数据
    CalibrationContext context;
    context.modelPoints = &modelPointsInWorldCoordinates;
    context.pixelPointsList = &listPixelPointsInPixelCoordinates;
    context.numViews = numViews;

    fdf_params = gsl_multifit_nlinear_default_parameters();
    fdf.f = [](const gsl_vector* x, void* params, gsl_vector* f) -> int {
        // 恢复上下文
        CalibrationContext* ctx = static_cast<CalibrationContext*>(params);

        float alpha = static_cast<float>(gsl_vector_get(x, 0));
        float gamma = static_cast<float>(gsl_vector_get(x, 1));
        float u0    = static_cast<float>(gsl_vector_get(x, 2));
        float beta  = static_cast<float>(gsl_vector_get(x, 3));
        float v0    = static_cast<float>(gsl_vector_get(x, 4));
        float k1    = static_cast<float>(gsl_vector_get(x, 5));
        float k2    = static_cast<float>(gsl_vector_get(x, 6));

        Eigen::Matrix3f iMat = MakeIntrinsicMatrix(alpha, beta, gamma, u0, v0);
        DistortFunctionBrownConrady distortFunction{ k1, k2 };

        // 全局残差索引
        size_t residualIdx = 0;

        for (size_t i = 0; i < ctx->numViews; ++i) {
            size_t base = 7 + i * 6;
            Eigen::Vector3f rVec{
                static_cast<float>(gsl_vector_get(x, base + 0)),
                static_cast<float>(gsl_vector_get(x, base + 1)),
                static_cast<float>(gsl_vector_get(x, base + 2)),
            };
            Eigen::Vector3f tVec{
                static_cast<float>(gsl_vector_get(x, base + 3)),
                static_cast<float>(gsl_vector_get(x, base + 4)),
                static_cast<float>(gsl_vector_get(x, base + 5)),
            };
            Eigen::Matrix3f rMat = Rodrigues(rVec);

            Eigen::Matrix2Xf img_reproj = Project(
                *(ctx->modelPoints),
                iMat, rMat, tVec, distortFunction
            );

            const Eigen::Matrix2Xf& img_obs = (*ctx->pixelPointsList)[i].topRows(2);

            // 计算差值
            img_reproj -= img_obs;

            // 批量填充到 f 中
            for (int k = 0; k < img_reproj.cols(); ++k) {
                gsl_vector_set(f, residualIdx++, img_reproj(0, k)); // x 误差
                gsl_vector_set(f, residualIdx++, img_reproj(1, k)); // y 误差
            }
        }

        return GSL_SUCCESS;
    };  // 残差函数
    fdf.df = nullptr;  // 雅可比函数 (如果设为 NULL，GSL 会使用有限差分近似)
    fdf.fvv = nullptr;  // 不使用测地线加速 (二阶导数)
    fdf.n = n;
    fdf.p = p;
    fdf.params = &context;

    // 分配求解器空间 (Trust Region - Levenberg Marquardt)
    T = gsl_multifit_nlinear_trust;
    w = gsl_multifit_nlinear_alloc(T, &fdf_params, n, p);

    // 初始化求解器
    gsl_multifit_nlinear_init(x, &fdf, w);

    // 3. 执行优化迭代
    int status;
    size_t iter = 0;
    const size_t max_iter = 100;
    const double xtol = 1e-4;  // 参数步长容差 (注意: gsl 通常用 double)
    const double gtol = 1e-4;  // 梯度容差
    const double ftol = 1e-4;  // 函数值容差

    int info;

    // 计算初始误差 (log)
    double initial_chi = gsl_blas_dnrm2(w->f);
    double initial_rmse = std::sqrt(initial_chi * initial_chi / n);
    std::cout << "[GSL] Initial RMSE: " << initial_rmse << std::endl;

    do {
        iter++;
        status = gsl_multifit_nlinear_iterate(w);

        if (status) {
            break;  // 发生错误
        }

        // 检查收敛条件
        status = gsl_multifit_nlinear_test(xtol, gtol, ftol, &info, w);

        if (iter % 10 == 0) {
            std::cout << "[GSL] Iter " << iter << " RMSE: "
                << std::sqrt(gsl_blas_dnrm2(w->f) * gsl_blas_dnrm2(w->f) / n)
                << std::endl;
        }

    } while (status == GSL_CONTINUE && iter < max_iter);

    // 4. 提取结果
    gsl_vector* x_opt = w->x; // 最优解
    intrinsicMatrix << static_cast<float>(gsl_vector_get(x_opt, 0)),
                       static_cast<float>(gsl_vector_get(x_opt, 1)),
                       static_cast<float>(gsl_vector_get(x_opt, 2)),
                       0.0f,
                       static_cast<float>(gsl_vector_get(x_opt, 3)),
                       static_cast<float>(gsl_vector_get(x_opt, 4)),
                       0.0f,
                       0.0f,
                       1.0f;
    radialDistortionCoeffcients << static_cast<float>(gsl_vector_get(x_opt, 5)),
                                   static_cast<float>(gsl_vector_get(x_opt, 6));

    double final_chi = gsl_blas_dnrm2(w->f);
    double final_rmse = std::sqrt(final_chi * final_chi / n);
    std::cout << "[GSL] Final RMSE: " << final_rmse << " (Status: " << gsl_strerror(status) << ")" << std::endl;

    // 清理内存
    gsl_multifit_nlinear_free(w);
    gsl_vector_free(x);

    std::cout << "[Info] Linear initialization finished.\n";
}

// 从 model.txt 读取数据
// 文件格式：每行 x y z
Eigen::Matrix3Xf LoadModelData(const char* pathData) {
    std::ifstream fs(pathData);
    if (!fs.is_open()) {
        throw std::runtime_error(std::string("Failed to open model file: ") + pathData);
    }

    std::vector<float> coords;

    {
        float valCoord;
        while (fs >> valCoord) {
            coords.push_back(valCoord);
        }
    }

    if (coords.empty() || coords.size() % 3 != 0) {
        throw std::runtime_error("Model file data format error or empty.");
    }

    auto numPoints = coords.size() / 3;
    Eigen::Matrix3Xf modelPoints(3, numPoints);

    for (int i = 0; i < numPoints; ++i) {
        modelPoints(0, i) = coords[3 * i + 0];  // X
        modelPoints(1, i) = coords[3 * i + 1];  // Y
        modelPoints(2, i) = 1.0f;               // W
    }

    std::cout << "Loaded " << numPoints << " points from " << pathData << std::endl;
    return modelPoints;
}

// 从 left*.txt 读取数据
// 参数 model 仅用于校验点数是否一致
std::vector<Eigen::Matrix3Xf> LoadPixelData(const Eigen::Matrix3Xf& model) {
    std::vector<Eigen::Matrix3Xf> pixelList;
    auto expectedPoints = model.cols();

    // 我们假设文件名为 left01.txt 到 left14.txt
    for (int i = 1; i <= 14; ++i) {
        if (i == 10) {
            continue;  // OpenCV 没有给提供 left10.jpg，因此也没有相应的 left10.txt
        }
        std::ostringstream ss;
        ss << "left" << std::setw(2) << std::setfill('0') << i << ".txt";
        std::string filename = ss.str();
        std::ifstream fs(filename);
        if (!fs.is_open()) {
            std::cout << "Skipping missing file: " << filename << std::endl;
            continue;
        }

        std::vector<float> coords;

        {
            float valCoord;
            while (fs >> valCoord) {
                coords.push_back(valCoord);
            }
        }

        if (coords.size() % 2 != 0) {
            std::cerr << "Warning: File " << filename << " has incomplete data points. Skipped." << std::endl;
            continue;
        }

        int numPoints = coords.size() / 2;
        if (numPoints != expectedPoints) {
            std::cerr << "Warning: File " << filename << " has " << numPoints
                      << " points, but model has " << expectedPoints << ". Skipped." << std::endl;
            continue;
        }

        Eigen::Matrix3Xf pixelPoints(3, numPoints);
        for (int k = 0; k < numPoints; ++k) {
            pixelPoints(0, k) = coords[2 * k + 0]; // u
            pixelPoints(1, k) = coords[2 * k + 1]; // v
            pixelPoints(2, k) = 1.0f;              // w
        }

        pixelList.push_back(pixelPoints);
        std::cout << "Loaded " << filename << std::endl;
    }

    if (pixelList.empty()) {
        throw std::runtime_error("No valid pixel data files found (left*.txt).");
    }

    return pixelList;
}

int main() {
    std::cout << "Starting Zhang's Calibration C++ implementation...\n";

    try {
        // 读取真实模型数据
        Eigen::Matrix3Xf modelPointsInWorldCoordinates = LoadModelData("model.txt");

        // 读取真实像素数据
        std::vector<Eigen::Matrix3Xf> listPixelPointsInPixelCoordinates = LoadPixelData(modelPointsInWorldCoordinates);

        std::vector<Eigen::Matrix3f> listHomography;
        std::cout << "Inferring Homographies...\n";
        for (const Eigen::Matrix3Xf& pixelPointsInPixelCoordinates : listPixelPointsInPixelCoordinates) {
            listHomography.push_back(InferHomography(modelPointsInWorldCoordinates, pixelPointsInPixelCoordinates));
        }

        Eigen::Matrix3f intrinsicMatrix;
        Eigen::Vector2f radialDistortionCoeffcients;

        std::cout << "Extracting Intrinsics...\n";
        ExtractIntrinsicParams(listHomography, modelPointsInWorldCoordinates, listPixelPointsInPixelCoordinates, intrinsicMatrix, radialDistortionCoeffcients);

        std::cout << "------------------------------------------\n"
                  << "Estimated Intrinsic Matrix (K):\n" << intrinsicMatrix << std::endl
                  << "Estimated Radial Distortion Coefficients:\n" << radialDistortionCoeffcients.transpose() << std::endl
                  << "------------------------------------------\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
