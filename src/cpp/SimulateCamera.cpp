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
#include <gsl/gsl_errno.h>

// ==========================================
//在此处控制是否校准 Gamma (Skew)
// 1: 校准 Gamma
// 0: 强制 Gamma = 0.0
#define CALIBRATE_GAMMA 0
// ==========================================

#define CONSTANT_PI 3.14159265358979323846
#define ARG_INPUT
#define ARG_OUTPUT
#define ARG_INPUT_OUTPUT

// 每一个列向量都是一个点的齐次坐标
using Point3dMatrix = Eigen::Matrix3Xd;
// 每一个列向量都是一个点的非齐次坐标
using Point2dMatrix = Eigen::Matrix2Xd;
// 用来构造系数矩阵
namespace Eigen {
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using RowVector6d = Eigen::Matrix<double, 1, 6>;
    using MatrixX6d = Eigen::Matrix<double, Eigen::Dynamic, 6>;
    using Vector9d = Eigen::Matrix<double, 9, 1>;
    using MatrixX9d = Eigen::Matrix<double, Eigen::Dynamic, 9>;
}

// [修改] float -> double
Eigen::Matrix3d MakeIntrinsicMatrix(double alpha, double beta, double gamma, double u0, double v0) {
    Eigen::Matrix3d matInParams;  // 内参矩阵
    matInParams << alpha, gamma, u0,
                   0.0,   beta,  v0,
                   0.0,   0.0,   1.0;
    return matInParams;
}

Eigen::Matrix3d Rodrigues(const Eigen::Vector3d& rVec) {
    double angle = rVec.norm();
    if (angle < 1e-8) {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, rVec.normalized()).toRotationMatrix();
}

Eigen::Vector3d InvRodrigues(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd angleAxis(R);
    return angleAxis.angle() * angleAxis.axis();
}

void Homo2Nonhomo(ARG_INPUT const Eigen::Matrix3Xd& homo, ARG_OUTPUT Eigen::Matrix2Xd& nonhomo) {
    if (homo.rows() != 3) {
        throw std::runtime_error("Homo2Nonhomo: homo must be 3xN.");
    }
    nonhomo.resize(2, homo.cols());
    Eigen::Array<double, 1, Eigen::Dynamic> denom = homo.row(2).array();
    denom = (denom.abs() < 1e-12).select(1e-12, denom);
    nonhomo.row(0).array() = homo.row(0).array() / denom;
    nonhomo.row(1).array() = homo.row(1).array() / denom;
}

void Nonhomo2Homo(ARG_INPUT const Eigen::Matrix2Xd& nonhomo, ARG_OUTPUT Eigen::Matrix3Xd& homo) {
    if (nonhomo.rows() != 2) {
        throw std::runtime_error("Nonhomo2Homo: nonhomo must be 3xN.");
    }
    homo.resize(3, nonhomo.cols());
    homo.row(0).array() = nonhomo.row(0).array();
    homo.row(1).array() = nonhomo.row(1).array();
    homo.row(2) = Eigen::RowVectorXd::Ones( nonhomo.cols() );
}

Eigen::Matrix3d IsotropicScalingNormalize(ARG_INPUT_OUTPUT Eigen::Matrix3Xd& points) {
    if (points.rows() != 3) {
        throw std::runtime_error("IsotropicScalingNormalize: points must be 3xN.");
    }
    static const double tiny = 1e-8;

    double centroidX = points.row(0).mean();
    double centroidY = points.row(1).mean();

    Eigen::Matrix2Xd centeredPoints = points.topRows(2);
    centeredPoints.row(0).array() -= centroidX;
    centeredPoints.row(1).array() -= centroidY;
    Eigen::VectorXd distances = centeredPoints.colwise().norm();
    double meanDistance = distances.mean();

    double scale = (meanDistance < tiny) ? 1.0 : (std::sqrt(2.0) / meanDistance);

    Eigen::Matrix3d sMat;
    sMat << scale,  0.0,   -scale * centroidX,
            0.0,    scale, -scale * centroidY,
            0.0,    0.0,   1.0;

    points = sMat * points;
    return sMat;
}

struct DistortFunction {  // 基类
    /**
     * @param points: 理想的（无畸变的）像素点坐标，归一化图像坐标系
     */
    virtual void Distort(ARG_INPUT_OUTPUT Eigen::Matrix3Xd& points) const {}

    virtual void DistortPoint(ARG_INPUT_OUTPUT Eigen::Vector2d& point) const {}
};

/**
 * 使用简化的 Brown-Conrady 畸变模型（只有2个径向畸变系数，没有切向畸变系数）
 */
struct DistortFunctionBrownConrady : DistortFunction {
    double k1, k2;

    DistortFunctionBrownConrady(double _k1, double _k2) : k1(_k1), k2(_k2) {}

    virtual void Distort(ARG_INPUT_OUTPUT Eigen::Matrix3Xd& points) const override {
        if (points.rows() != 3) {
            throw std::runtime_error("DistortFunctionBrownConrady::Distort: points must be 3xN.");
        }
        Eigen::Matrix2Xd nonhomo;
        Homo2Nonhomo(points, nonhomo);
        for (int i = 0; i < nonhomo.cols(); ++i) {
            Eigen::Vector2d point = nonhomo.col(i);
            DistortPoint(point);
            nonhomo.col(i) = point;
        }
        points.row(0) = nonhomo.row(0);
        points.row(1) = nonhomo.row(1);
        points.row(2).setOnes();
    }

    virtual void DistortPoint(ARG_INPUT_OUTPUT Eigen::Vector2d& point) const override {
        double r2 = point.squaredNorm();
        double coeff = 1.0 + k1 * r2 + k2 * r2 * r2;
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
auto Project(const Eigen::Matrix3Xd& modelPointsInWorldCoordinates,
             const Eigen::Matrix3d& iMat, const Eigen::Matrix3d& rMat, const Eigen::Vector3d& tVec,
             const DistortFunction& distortFunction) {
    if (modelPointsInWorldCoordinates.rows() != 3) {
        throw std::runtime_error("Project: modelPointsInWorldCoordinates must be 3xN.");
    }
    Eigen::Matrix3d rtMat;
    Eigen::Matrix3Xd modelPointsInCameraCoordinates;
    Eigen::Matrix3Xd pixelPointsInImageCoordinates;
    Eigen::Matrix2Xd pixelPointsInPixelCoordinates;

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

Eigen::Matrix3d InferHomography(ARG_INPUT const Eigen::Matrix3Xd& modelPointsInWorldCoordinates,
                                ARG_INPUT const Eigen::Matrix3Xd& pixelPointsInPixelCoordinates) {
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
    Eigen::Matrix3Xd modelHomo = modelPointsInWorldCoordinates;
    Eigen::Matrix3Xd pixelHomo = pixelPointsInPixelCoordinates;
    Eigen::Matrix3d T_model = IsotropicScalingNormalize(modelHomo);
    Eigen::Matrix3d T_pixel = IsotropicScalingNormalize(pixelHomo);

    int nColsModel = modelHomo.cols();
    // 2. 构造矩阵 matL (2N x 9)
    Eigen::MatrixX9d matL(2 * nColsModel, 9);
    matL.setZero();
    for (int i = 0; i < nColsModel; ++i) {
        double x = modelHomo(0, i);
        double y = modelHomo(1, i);
        double w = modelHomo(2, i);
        double u = pixelHomo(0, i);
        double v = pixelHomo(1, i);
        matL.row(2 * i)     << x, y, w, 0, 0, 0, -u * x, -u * y, -u * w;
        matL.row(2 * i + 1) << 0, 0, 0, x, y, w, -v * x, -v * y, -v * w;
    }

    if (matL.rows() != 2 * nColsModel || matL.cols() != 9) {
        throw std::runtime_error("InferHomography: matL shape mismatch.");
    }

    // 3. SVD 求解，取最小奇异值对应的右奇异向量
    Eigen::JacobiSVD<Eigen::MatrixX9d> svd(matL, Eigen::ComputeThinV);
    Eigen::Vector9d h = svd.matrixV().col(svd.matrixV().cols() - 1);

    if (h.size() != 9) {
        throw std::runtime_error("InferHomography: SVD solution vector size mismatch.");
    }

    // 4. 重构 H_norm 并去归一化
    Eigen::Matrix3d H_norm;
    H_norm << h(0), h(1), h(2),
              h(3), h(4), h(5),
              h(6), h(7), h(8);

    Eigen::Matrix3d H = T_pixel.inverse() * H_norm * T_model;

    if (std::abs(H(2, 2)) > 1e-8) {
        H /= H(2, 2);
    }

    return H;
}

struct CalibrationContext {
    const Eigen::Matrix3Xd* modelPoints;
    const std::vector<Eigen::Matrix3Xd>* pixelPointsList;
    size_t numViews;
};

void ExtractIntrinsicParams(ARG_INPUT const std::vector<Eigen::Matrix3d>& listHomography,
                            ARG_INPUT const Eigen::Matrix3Xd& modelPointsInWorldCoordinates,
                            ARG_INPUT const std::vector<Eigen::Matrix3Xd>& listPixelPointsInPixelCoordinates,
                            ARG_OUTPUT Eigen::Matrix3d& intrinsicMatrix,
                            ARG_OUTPUT Eigen::Vector2d& radialDistortionCoeffcients) {
    if (modelPointsInWorldCoordinates.rows() != 3) {
        throw std::runtime_error("ExtractIntrinsicParams: modelPointsInWorldCoordinates must be 3xN.");
    }
    if (listHomography.size() != listPixelPointsInPixelCoordinates.size()) {
        throw std::runtime_error("ExtractIntrinsicParams: listHomography and listPixelPointsInPixelCoordinates size mismatch.");
    }

    auto numViews = listHomography.size();
    const size_t numPointsPerView = modelPointsInWorldCoordinates.cols();

    if (numViews < 3) {
        std::cerr << "Warning: At least 3 views are required for calibration.\n";
    }

    std::vector<Eigen::Vector3d> rVecs;
    std::vector<Eigen::Vector3d> tVecs;

    rVecs.reserve(numViews);
    tVecs.reserve(numViews);

    {  // make_initial_guess
        // v_ij 函数的 lambda 实现，返回 1x6 向量
        const auto create_v_ij = [](const Eigen::Matrix3d& matHomography, int i, int j) -> Eigen::RowVector6d {
            Eigen::RowVector6d v;
            v(0) = matHomography(0, i) * matHomography(0, j);
            v(1) = matHomography(0, i) * matHomography(1, j) + matHomography(1, i) * matHomography(0, j);
            v(2) = matHomography(1, i) * matHomography(1, j);
            v(3) = matHomography(2, i) * matHomography(0, j) + matHomography(0, i) * matHomography(2, j);
            v(4) = matHomography(2, i) * matHomography(1, j) + matHomography(1, i) * matHomography(2, j);
            v(5) = matHomography(2, i) * matHomography(2, j);
            return v;
        };

        // 构造系数矩阵
        Eigen::MatrixX6d matV(2 * numViews, 6);
        for (decltype(numViews) k = 0; k < numViews; ++k) {
            const Eigen::Matrix3d& matHomography = listHomography[k];
            matV.row(2 * k) = create_v_ij(matHomography, 0, 1);
            matV.row(2 * k + 1) = create_v_ij(matHomography, 0, 0) - create_v_ij(matHomography, 1, 1);
        }

        if (matV.rows() != 2 * numViews || matV.cols() != 6) {
             throw std::runtime_error("ExtractIntrinsicParams: matV shape mismatch.");
        }

        Eigen::JacobiSVD<Eigen::MatrixX6d> svd(matV, Eigen::ComputeThinV);
        // b = [B11, B12, B22, B13, B23, B33]
        Eigen::Vector6d b = svd.matrixV().col(svd.matrixV().cols() - 1);

        if (b.size() != 6) {
             throw std::runtime_error("ExtractIntrinsicParams: b vector size mismatch.");
        }

        if (b(0) < 0) {  // 确保 B11 为正
            b = -b;
        }

        {  // 计算基本矩阵、中间变量
            double B11 = b(0);
            double B12 = b(1);
            double B22 = b(2);
            double B13 = b(3);
            double B23 = b(4);
            double B33 = b(5);

            double mid1 = B12 * B13 - B11 * B23;
            double mid2 = B11 * B22 - B12 * B12;
            double v0 = mid1 / mid2;
            double lambda = B33 - (B13 * B13 + v0 * mid1) / B11;
            double alpha2 = lambda / B11;
            double alpha = std::sqrt(alpha2);
            double beta = std::sqrt(lambda * B11 / mid2);
            double gamma = -B12 * alpha2 * beta / lambda;
            double u0 = gamma * v0 / beta - B13 * alpha2 / lambda;

            intrinsicMatrix <<  alpha, gamma, u0,
                                0.0,   beta,  v0,
                                0.0,   0.0,   1.0;
            radialDistortionCoeffcients << 0.0, 0.0;
        }

#if CALIBRATE_GAMMA == 0
        // 如果禁用 Gamma，在初始猜测中强制置零
        intrinsicMatrix(0, 1) = 0.0;
#endif

        Eigen::Matrix3d invIntrinsicMatrix = intrinsicMatrix.inverse();

        const auto approximate_rotation_matrix = [](Eigen::Matrix3d& mat) -> void {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            const Eigen::Matrix3d& V = svd.matrixV();
            mat = U * V.transpose();

            if (mat.determinant() < 0) {
                U.col(2) *= -1;
                mat = U * V.transpose();
            }
        };

        for (const Eigen::Matrix3d& matHomography : listHomography) {
            const Eigen::Vector3d& h1 = matHomography.col(0);
            const Eigen::Vector3d& h2 = matHomography.col(1);
            const Eigen::Vector3d& h3 = matHomography.col(2);
            Eigen::Vector3d invKh1 = invIntrinsicMatrix * h1;
            Eigen::Vector3d invKh2 = invIntrinsicMatrix * h2;
            Eigen::Vector3d invKh3 = invIntrinsicMatrix * h3;
            double lambda1 = 1.0 / invKh1.norm();
            Eigen::Vector3d r1 = lambda1 * invKh1;
            Eigen::Vector3d r2 = lambda1 * invKh2;
            Eigen::Vector3d r3 = r1.cross(r2);
            Eigen::Vector3d t = lambda1 * invKh3;
            Eigen::Matrix3d matR;
            matR.col(0) = r1;
            matR.col(1) = r2;
            matR.col(2) = r3;
            approximate_rotation_matrix(matR);
            rVecs.push_back(InvRodrigues(matR));
            tVecs.push_back(t);
        }
    }

    // 参数数量计算
    // 基础内参：alpha, beta, u0, v0 (4个)
    // 可选内参：gamma (1个)
    // 畸变：k1, k2 (2个)
    // 外参：6 * numViews
#if CALIBRATE_GAMMA == 1
    const size_t numIntrinsicParams = 7; // alpha, gamma, u0, beta, v0, k1, k2
#else
    const size_t numIntrinsicParams = 6; // alpha, u0, beta, v0, k1, k2
#endif
    const size_t p = numIntrinsicParams + 6 * numViews;

    // 残差数量
    const size_t n = 2 * numPointsPerView * numViews;

    gsl_vector* x;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params;
    const gsl_multifit_nlinear_type* T;
    gsl_multifit_nlinear_workspace* w;

    // 初始化参数向量 x
    x = gsl_vector_alloc(p);

#if CALIBRATE_GAMMA == 1
    gsl_vector_set(x, 0, intrinsicMatrix(0, 0)); // alpha
    gsl_vector_set(x, 1, intrinsicMatrix(0, 1)); // gamma
    gsl_vector_set(x, 2, intrinsicMatrix(0, 2)); // u0
    gsl_vector_set(x, 3, intrinsicMatrix(1, 1)); // beta
    gsl_vector_set(x, 4, intrinsicMatrix(1, 2)); // v0
    gsl_vector_set(x, 5, 0.0);                   // k1
    gsl_vector_set(x, 6, 0.0);                   // k2
#else
    gsl_vector_set(x, 0, intrinsicMatrix(0, 0)); // alpha
    gsl_vector_set(x, 1, intrinsicMatrix(0, 2)); // u0
    gsl_vector_set(x, 2, intrinsicMatrix(1, 1)); // beta
    gsl_vector_set(x, 3, intrinsicMatrix(1, 2)); // v0
    gsl_vector_set(x, 4, 0.0);                   // k1
    gsl_vector_set(x, 5, 0.0);                   // k2
#endif

    for (size_t i = 0; i < numViews; ++i) {
        size_t base = numIntrinsicParams + i * 6;
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

    // 启用 Marquardt 缩放 (Levenberg-Marquardt scaling)
    fdf_params.scale = gsl_multifit_nlinear_scale_more;
    // 指定使用 Levenberg-Marquardt 算法
    fdf_params.trs = gsl_multifit_nlinear_trs_lm;

    fdf.f = [](const gsl_vector* x, void* params, gsl_vector* f) -> int {
        // 恢复上下文
        CalibrationContext* ctx = static_cast<CalibrationContext*>(params);

#if CALIBRATE_GAMMA == 1
        double alpha = gsl_vector_get(x, 0);
        double gamma = gsl_vector_get(x, 1);
        double u0    = gsl_vector_get(x, 2);
        double beta  = gsl_vector_get(x, 3);
        double v0    = gsl_vector_get(x, 4);
        double k1    = gsl_vector_get(x, 5);
        double k2    = gsl_vector_get(x, 6);
        const size_t numIntrinsics = 7;
#else
        double alpha = gsl_vector_get(x, 0);
        double gamma = 0.0;
        double u0    = gsl_vector_get(x, 1);
        double beta  = gsl_vector_get(x, 2);
        double v0    = gsl_vector_get(x, 3);
        double k1    = gsl_vector_get(x, 4);
        double k2    = gsl_vector_get(x, 5);
        const size_t numIntrinsics = 6;
#endif

        Eigen::Matrix3d iMat = MakeIntrinsicMatrix(alpha, beta, gamma, u0, v0);
        DistortFunctionBrownConrady distortFunction{ k1, k2 };

        size_t residualIdx = 0;

        for (size_t i = 0; i < ctx->numViews; ++i) {
            size_t base = numIntrinsics + i * 6;
            Eigen::Vector3d rVec{
                gsl_vector_get(x, base + 0),
                gsl_vector_get(x, base + 1),
                gsl_vector_get(x, base + 2)
            };
            Eigen::Vector3d tVec{
                gsl_vector_get(x, base + 3),
                gsl_vector_get(x, base + 4),
                gsl_vector_get(x, base + 5)
            };
            Eigen::Matrix3d rMat = Rodrigues(rVec);

            Eigen::Matrix2Xd img_reproj = Project(
                *(ctx->modelPoints),
                iMat, rMat, tVec, distortFunction
            );

            const Eigen::Matrix2Xd& img_obs = (*ctx->pixelPointsList)[i].topRows(2);

            img_reproj -= img_obs;

            for (int k = 0; k < img_reproj.cols(); ++k) {
                gsl_vector_set(f, residualIdx++, img_reproj(0, k));
                gsl_vector_set(f, residualIdx++, img_reproj(1, k));
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

    // 关闭 GSL 的默认错误处理程序（防止直接崩溃）
    gsl_set_error_handler_off();

    // 3. 执行优化迭代
    int status;
    size_t iter = 0;
    const size_t max_iter = 100;
    const double xtol = 1e-8;
    const double gtol = 1e-8;
    const double ftol = 1e-8;

    int info;

    double initial_chi = gsl_blas_dnrm2(w->f);
    double initial_rmse = std::sqrt(initial_chi * initial_chi / n);
    std::cout << "[GSL] Initial RMSE: " << initial_rmse << std::endl;

    do {
        iter++;
        status = gsl_multifit_nlinear_iterate(w);

        // 如果出错，打印警告但不要退出程序，保留当前结果
        if (status) {
            std::cerr << "[GSL Warning] Solver stopped early at iter " << iter
                      << ": " << gsl_strerror(status) << std::endl;
            break;
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
#if CALIBRATE_GAMMA == 1
    intrinsicMatrix << gsl_vector_get(x_opt, 0),
                       gsl_vector_get(x_opt, 1),
                       gsl_vector_get(x_opt, 2),
                       0.0,
                       gsl_vector_get(x_opt, 3),
                       gsl_vector_get(x_opt, 4),
                       0.0,
                       0.0,
                       1.0;
    radialDistortionCoeffcients << gsl_vector_get(x_opt, 5),
                                   gsl_vector_get(x_opt, 6);
#else
    intrinsicMatrix << gsl_vector_get(x_opt, 0),
                       0.0,
                       gsl_vector_get(x_opt, 1),
                       0.0,
                       gsl_vector_get(x_opt, 2),
                       gsl_vector_get(x_opt, 3),
                       0.0,
                       0.0,
                       1.0;
    radialDistortionCoeffcients << gsl_vector_get(x_opt, 4),
                                   gsl_vector_get(x_opt, 5);
#endif

    double final_chi = gsl_blas_dnrm2(w->f);
    double final_rmse = std::sqrt(final_chi * final_chi / n);
    std::cout << "[GSL] Final RMSE: " << final_rmse << " (Status: " << gsl_strerror(status) << ", n_iter: " << iter << ")" << std::endl;

    // 清理内存
    gsl_multifit_nlinear_free(w);
    gsl_vector_free(x);

    std::cout << "[Info] Linear initialization finished.\n";
}

// 从 model.txt 读取数据
// 文件格式：每行 x y z
Eigen::Matrix3Xd LoadModelData(const char* pathData) {
    std::ifstream fs(pathData);
    if (!fs.is_open()) {
        throw std::runtime_error(std::string("Failed to open model file: ") + pathData);
    }

    std::vector<double> coords;
    double valCoord;
    while (fs >> valCoord) {
        coords.push_back(valCoord);
    }

    if (coords.empty() || coords.size() % 3 != 0) {
        throw std::runtime_error("Model file data format error or empty.");
    }

    auto numPoints = coords.size() / 3;
    Eigen::Matrix3Xd modelPoints(3, numPoints);

    for (int i = 0; i < numPoints; ++i) {
        modelPoints(0, i) = coords[3 * i + 0];
        modelPoints(1, i) = coords[3 * i + 1];
        modelPoints(2, i) = 1.0;
    }

    std::cout << "Loaded " << numPoints << " points from " << pathData << std::endl;
    return modelPoints;
}

// 从 left*.txt 读取数据
// 参数 model 仅用于校验点数是否一致
std::vector<Eigen::Matrix3Xd> LoadPixelData(const Eigen::Matrix3Xd& model) {
    std::vector<Eigen::Matrix3Xd> pixelList;
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

        std::vector<double> coords;
        double valCoord;
        while (fs >> valCoord) {
            coords.push_back(valCoord);
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

        Eigen::Matrix3Xd pixelPoints(3, numPoints);
        for (int k = 0; k < numPoints; ++k) {
            pixelPoints(0, k) = coords[2 * k + 0];
            pixelPoints(1, k) = coords[2 * k + 1];
            pixelPoints(2, k) = 1.0;
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
        Eigen::Matrix3Xd modelPointsInWorldCoordinates = LoadModelData("model.txt");

        // 读取真实像素数据
        std::vector<Eigen::Matrix3Xd> listPixelPointsInPixelCoordinates = LoadPixelData(modelPointsInWorldCoordinates);

        std::vector<Eigen::Matrix3d> listHomography;
        std::cout << "Inferring Homographies...\n";
        for (const Eigen::Matrix3Xd& pixelPointsInPixelCoordinates : listPixelPointsInPixelCoordinates) {
            listHomography.push_back(InferHomography(modelPointsInWorldCoordinates, pixelPointsInPixelCoordinates));
        }

        Eigen::Matrix3d intrinsicMatrix;
        Eigen::Vector2d radialDistortionCoeffcients;

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
