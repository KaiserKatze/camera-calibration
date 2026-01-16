#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#define CONSTANT_PI 3.14159265358979323846f
#define ARG_INPUT
#define ARG_OUTPUT
#define ARG_INPUT_OUTPUT

// 每一个列向量都是一个点的齐次坐标
using Point3dMatrix = Eigen::Matrix3Xf;
// 每一个列向量都是一个点的非齐次坐标
using Point2dMatrix = Eigen::Matrix2Xf;

struct InParams {
    float alpha;
    float beta;
    float gamma;
    float u0;
    float v0;

    Eigen::Matrix3f GetMatrix() const {
        Eigen::Matrix3f inParamMat;  // 内参矩阵
        inParamMat << alpha, gamma, u0,
                      0.0f,  beta,  v0,
                      0.0f,  0.0f,  1.0f;
        return inParamMat;
    }
};

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

void Homo2Nonhomo(ARG_INPUT const Eigen::Matrix3Xf& homo, ARG_OUTPUT Eigen::Matrix2Xf& nonhomo) {
    Eigen::ArrayXf denom = homo.row(2).array();
    denom = (denom.abs() < 1e-12).select(1e-12, denom); // 如果绝对值小于阈值，设为 1e-12
    nonhomo.row(0).array() = homo.row(0).array() / denom;
    nonhomo.row(1).array() = homo.row(1).array() / denom;
}

void Nonhomo2Homo(ARG_INPUT const Eigen::Matrix2Xf& nonhomo, ARG_OUTPUT Eigen::Matrix3Xf& homo) {
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
    virtual void distort(ARG_INPUT_OUTPUT Eigen::Matrix3Xf& points) const {}  // 默认无畸变
};

/**
 * 使用简化的 Brown-Conrady 畸变模型（只有2个径向畸变系数，没有切向畸变系数）
 */
struct DistortFunctionBrownConrady : DistortFunction {
    float k1, k2;

    DistortFunctionBrownConrady(float _k1, float _k2) : k1(_k1), k2(_k2) {}

    /**
     * @param points: 理想的（无畸变的）像素点坐标，归一化图像坐标系中
     */
    virtual void distort(ARG_INPUT_OUTPUT Eigen::Matrix3Xf& points) const override {
        Eigen::Matrix2Xf nonhomo;
        Homo2Nonhomo(points, nonhomo);
        for (int i = 0; i < nonhomo.cols(); ++i) {
            float x = nonhomo(0, i);
            float y = nonhomo(1, i);
            float r2 = x * x + y * y;
            float coeff = 1.0f + k1 * r2 + k2 * r2 * r2;
            nonhomo(0, i) = x * coeff;
            nonhomo(1, i) = y * coeff;
        }
        points.row(0) = nonhomo.row(0);
        points.row(1) = nonhomo.row(1);
    }
};

auto Project(const Eigen::Matrix3Xf& modelPointsInWorldCoordinates,
             const Eigen::Matrix3f& iMat, const Eigen::Matrix3f& rMat, const Eigen::Vector3f& tVec,
             const DistortFunction& distortFunction) {
    /**
     * 将模型点投影到像平面上，得到像素点的非齐次坐标
     *
     * @param modelPointsInWorldCoordinates: 模型点在世界坐标系中的齐次坐标（默认 Z 坐标为 0）
     * @param iMat: 相机内部参数矩阵
     * @param rMat: 旋转矩阵
     * @param tVec: 平移向量
     */
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
    distortFunction.distort(modelPointsInCameraCoordinates);
    // 将模型点投影到像平面，得到像素点在像平面坐标系上的齐次坐标
    pixelPointsInImageCoordinates = iMat * modelPointsInCameraCoordinates;
    // 归一化得到像素点的非齐次坐标
    Homo2Nonhomo(pixelPointsInImageCoordinates, pixelPointsInPixelCoordinates);
    return pixelPointsInPixelCoordinates;
}

// typedef Eigen::Matrix<float, 2, 1> Eigen::Vector2f


Eigen::Matrix3f InferHomography(ARG_INPUT const Eigen::Matrix3Xf& modelPointsInWorldCoordinates,
                                ARG_INPUT const Eigen::Matrix3Xf& pixelPointsInPixelCoordinates) {
    Eigen::Matrix3Xf modelHomo{ modelPointsInWorldCoordinates };
    Eigen::Matrix3Xf pixelHomo{ pixelPointsInPixelCoordinates };
    Eigen::Matrix3f sMatForModel = IsotropicScalingNormalize(modelHomo);
    Eigen::Matrix3f sMatForPixel = IsotropicScalingNormalize(pixelHomo);
    Eigen::Matrix3Xf zeros;
    Eigen::Matrix3f homography;
    // TODO 完成 infer_homography_without_radial_distortion 和 infer_homography_without_radial_distortion_with_isotropic_scaling 所做的工作
    return homography;
}

Eigen::Matrix3Xf LoadModelData(const char* pathData) {
    // 从硬盘上的某个文件读取模型点的坐标数据
}

std::vector<Eigen::Matrix3Xf> LoadPixelData(const char* pathData) {
    // 从硬盘上的某个文件读取像素点的坐标数据
}

void ExtractIntrinsicParams(ARG_INPUT std::vector<Eigen::Matrix3f>& listHomography,
                            ARG_INPUT Eigen::Matrix3Xf& modelPointsInWorldCoordinates,
                            ARG_INPUT std::vector<Eigen::Matrix3Xf>& listPixelPointsInPixelCoordinates,
                            ARG_OUTPUT Eigen::Matrix3f& intrinsicMatrix,
                            ARG_OUTPUT Eigen::Vector2f& radialDistortionCoeffcients) {
    // TODO 完成 extract_intrinsic_params_and_radial_distort_coeff_from_homography 的工作
}

int main() {
    Eigen::Matrix3Xf modelPointsInWorldCoordinates = LoadModelData("models.txt");
    std::vector<Eigen::Matrix3Xf> listPixelPointsInPixelCoordinates = LoadPixelData("pixels.txt");
    std::vector<Eigen::Matrix3f> listHomography;
    for (const Eigen::Matrix3Xf& pixelPointsInPixelCoordinates : listPixelPointsInPixelCoordinates) {
        listHomography.push_back(InferHomography(modelPointsInWorldCoordinates, pixelPointsInPixelCoordinates));
    }
    Eigen::Matrix3f intrinsicMatrix;
    Eigen::Vector2f radialDistortionCoeffcients;
    ExtractIntrinsicParams(listHomography, modelPointsInWorldCoordinates, listPixelPointsInPixelCoordinates, intrinsicMatrix, radialDistortionCoeffcients);
    std::cout << "Estimated Intrinsic Matrix:\n" << intrinsicMatrix
        << "\nEstimated Radial Distortion Coefficients:\n" << radialDistortionCoeffcients << '\n';
    return 0;
}
