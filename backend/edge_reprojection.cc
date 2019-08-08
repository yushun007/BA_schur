#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include "backend/vertex_pose.h"
#include "backend/vertex_point_xyz.h"
#include "backend/edge_reprojection.h"
#include "backend/eigen_types.h"

#include <iostream>

namespace myslam {
namespace backend {

/*    std::vector<std::shared_ptr<Vertex>> verticies_; // 该边对应的顶点
    VecX residual_;                 // 残差
    std::vector<MatXX> jacobians_;  // 雅可比，每个雅可比维度是 residual x vertex[i]
    MatXX information_;             // 信息矩阵
    VecX observation_;              // 观测信息
    */
//该函数是三元边 顶点为InveseDepth、T_World_From_Body1、T_World_From_Body2
void EdgeReprojection::ComputeResidual() {
    double inv_dep_i = verticies_[0]->Parameters()[0];//定义inversedepth

    VecX param_i = verticies_[1]->Parameters();//定义T_World_From_Body1即先验pose
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);//四元数旋转
    Vec3 Pi = param_i.head<3>();//平移向量

    VecX param_j = verticies_[2]->Parameters();//定义T_World_From_Body2后验pose
    Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Vec3 Pj = param_j.head<3>();

    Vec3 pts_camera_i = pts_i_ / inv_dep_i;//pts_i_为i相机下的像素坐标/pts_camera_i为相机归一化平面坐标

    Vec3 pts_imu_i = qic * pts_camera_i + tic;//将相机坐标系下的向量旋转平移到imu坐标系下
    Vec3 pts_w = Qi * pts_imu_i + Pi;//landmark世界坐标为pts_w
    Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);//j时刻imu坐标系下的landmark估计坐标
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);//j时刻camera坐标系下landmark的估计坐标

    double dep_j = pts_camera_j.z();
    residual_ = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();   /// J^t * J * delta_x = - J^t * r//重投影误差
//    residual_ = information_ * residual_;   // remove information here, we multi information matrix in problem solver
}
//camera到imu坐标系的转换
void EdgeReprojection::SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_) {
    qic = qic_;
    tic = tic_;
}

void EdgeReprojection::ComputeJacobians() {
    double inv_dep_i = verticies_[0]->Parameters()[0];

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    VecX param_j = verticies_[2]->Parameters();
    Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Vec3 Pj = param_j.head<3>();

    Vec3 pts_camera_i = pts_i_ / inv_dep_i;
    Vec3 pts_imu_i = qic * pts_camera_i + tic;
    Vec3 pts_w = Qi * pts_imu_i + Pi;
    Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    double dep_j = pts_camera_j.z();

    Mat33 Ri = Qi.toRotationMatrix();//i时刻的imu到世界旋转矩阵
    Mat33 Rj = Qj.toRotationMatrix();//j时刻的imu到世界旋转矩阵
    Mat33 ric = qic.toRotationMatrix();//camera到imu的旋转矩阵
    Mat23 reduce(2, 3);//误差矩阵
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);//误差对fcj的求导
        //[1./dep_j,0,-pts_camera_j(0)],[0,1/dep_j,-pts_camera_j[1]/dep_j*dep_j]
     //reduce = information_ * reduce;

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
    //i时刻fci对位移求导
    jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Sophus::SO3d::hat(pts_imu_i);
    //i时刻fci对角度增量的求导
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
    //根据求导链式法则误差对i时刻pose的雅克比为fci的导数乘以fci对各状态量的雅克比

    Eigen::Matrix<double, 2, 6> jacobian_pose_j;
    Eigen::Matrix<double, 3, 6> jaco_j;
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
    jacobian_pose_j.leftCols<6>() = reduce * jaco_j;

    Eigen::Vector2d jacobian_feature;//误差对imu与camera之间的外参的雅克比
    jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);

    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;
    jacobians_[2] = jacobian_pose_j;

    ///------------- check jacobians -----------------
//    {
//        std::cout << jacobians_[0] <<std::endl;
//        const double eps = 1e-6;
//        inv_dep_i += eps;
//        Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
//        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
//        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
//        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
//        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
//
//        Eigen::Vector2d tmp_residual;
//        double dep_j = pts_camera_j.z();
//        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
//        tmp_residual = information_ * tmp_residual;
//        std::cout <<"num jacobian: "<<  (tmp_residual - residual_) / eps <<std::endl;
//    }

}
//二元边误差计算,是两个顶点分别为landmark世界坐标点XYZ,和观测到landmark的camera的pose
void EdgeReprojectionXYZ::ComputeResidual() {
    Vec3 pts_w = verticies_[0]->Parameters();//landmark世界坐标点

    VecX param_i = verticies_[1]->Parameters();//相机pose
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);//旋转四元数
    Vec3 Pi = param_i.head<3>();//平移向量

    Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);//landmark_imu坐标
    Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);//landmark相机坐标

    double dep_i = pts_camera_i.z();//逆深度
    residual_ = (pts_camera_i / dep_i).head<2>() - obs_.head<2>();
}

void EdgeReprojectionXYZ::SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_) {
    qic = qic_;
    tic = tic_;
}

void EdgeReprojectionXYZ::ComputeJacobians() {

    Vec3 pts_w = verticies_[0]->Parameters();

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
    Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);

    double dep_i = pts_camera_i.z();

    Mat33 Ri = Qi.toRotationMatrix();
    Mat33 ric = qic.toRotationMatrix();
    Mat23 reduce(2, 3);
    reduce << 1. / dep_i, 0, -pts_camera_i(0) / (dep_i * dep_i),
        0, 1. / dep_i, -pts_camera_i(1) / (dep_i * dep_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * -Ri.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

    Eigen::Matrix<double, 2, 3> jacobian_feature;
    jacobian_feature = reduce * ric.transpose() * Ri.transpose();

    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;

}

void EdgeReprojectionPoseOnly::ComputeResidual() {
    VecX pose_params = verticies_[0]->Parameters();
    Sophus::SE3d pose(
        Qd(pose_params[6], pose_params[3], pose_params[4], pose_params[5]),
        pose_params.head<3>()
    );//将相机pose转换为李群

    Vec3 pc = pose * landmark_world_;
    pc = pc / pc[2];
    Vec2 pixel = (K_ * pc).head<2>() - observation_;
    // TODO:: residual_ = ????
    residual_ = pixel;
}

void EdgeReprojectionPoseOnly::ComputeJacobians() {
    // TODO implement jacobian here
}

}
}