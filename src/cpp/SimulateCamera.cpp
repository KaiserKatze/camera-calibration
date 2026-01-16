#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

struct InParams {
    float alpha;
    float beta;
    float u0;
    float v0;

    Eigen::Matrix3f GetMatrix() const {
        Eigen::Matrix3f inParamMat;  // 内参矩阵
        inParamMat << alpha, 0.0f, u0,
            0.0f, beta, v0,
            0.0f, 0.0f, 1.0f;
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

    Eigen::Matrix3f GetRotation() const {
        // 按照 X,Y,Z 的顺序计算
        Eigen::AngleAxisf rollAngle  (rx, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle (ry, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle   (rz, Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf q = yawAngle * pitchAngle * rollAngle;
        Eigen::Matrix3f rotationMatrix = q.matrix();
        return rotationMatrix;
    }

    Eigen::Matrix3f GetTranslation() const {
        Eigen::Matrix3f tVec;  // 平移向量
        tVec << tx, ty, tz;
        return tVec;
    }
};

int main() {
    float rx;
    float ry;
    float rz;

    rx = 3.14159f / 4.0f;
    ry = 3.14159f / 3.0f;
    rz = 3.14159f / 6.0f;
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d rotationMatrix = q.matrix();
    std::cout << "旋转矩阵1:\n" << rotationMatrix << std::endl;

    ExParams exParams{
        rx, ry, rz, 0.0f, 0.0f, 0.0f,
    };
    std::cout << "旋转矩阵2:\n" << exParams.GetRotation() << std::endl;
    return 0;
}
