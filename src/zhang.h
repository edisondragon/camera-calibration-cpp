// ============================================================================
//  zhang.h  --  张氏标定法（Zhang's calibration）接口
//
//  实现的完整流程：
//    1. DLT + Hartley 归一化     求每张图的单应矩阵 H
//    2. 正交约束 V*b = 0         求 B 矩阵
//    3. 闭式公式                 从 B 反解内参 K
//    4. SVD 正交化               从 H 和 K 求外参 R, t
//    5. 线性最小二乘             求径向畸变 k1, k2
//    6. Levenberg-Marquardt      重投影误差精化
// ============================================================================
#pragma once

#include "linalg.h"
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
//  基本类型
// ---------------------------------------------------------------------------
struct Point2 { double x = 0, y = 0; };
struct Point3 { double x = 0, y = 0, z = 0; };

// 一张标定图：板上各角点的世界坐标 + 对应的图像像素坐标
struct View {
    std::vector<Point3> obj;   // 板坐标（通常 Z = 0）
    std::vector<Point2> img;   // 检测到的像素坐标
};

// 标定结果
struct CalibResult {
    Mat K;                        // 3x3 内参
    double k1 = 0.0;              // 径向畸变
    double k2 = 0.0;
    std::vector<Mat>  R;          // 每张图的旋转
    std::vector<Vec>  t;          // 每张图的平移
    double rms = 0.0;             // 重投影误差 RMS（像素）
    bool   ok = false;
    std::string message;
};

// ---------------------------------------------------------------------------
//  阶段一：线性闭式解
// ---------------------------------------------------------------------------

// 用 DLT 求单应矩阵（含 Hartley 归一化）
Mat computeHomography(const std::vector<Point3>& obj,
                      const std::vector<Point2>& img);

// 由 B 的 6 个元素构造内参矩阵 K
Mat KfromB(const Vec& b);

// 从单应矩阵和外参求 R, t
void extrinsicsFromH(const Mat& H, const Mat& K, Mat& R, Vec& t);

// 求径向畸变（线性最小二乘）
void estimateDistortion(const std::vector<View>& views,
                        const std::vector<Mat>& H,
                        const Mat& K,
                        double& k1, double& k2);

// ---------------------------------------------------------------------------
//  阶段二：非线性精化
// ---------------------------------------------------------------------------

// 用 LM 最小化重投影误差
void refineLM(const std::vector<View>& views, CalibResult& res);

// ---------------------------------------------------------------------------
//  顶层接口
// ---------------------------------------------------------------------------
CalibResult calibrate(const std::vector<View>& views, bool doRefine = true);

// 计算重投影误差 RMS
double reprojectionRMS(const std::vector<View>& views, const CalibResult& res);
