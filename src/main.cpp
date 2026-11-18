// ============================================================================
//  main.cpp  --  命令行入口
//
//  用法:
//     zhang_calib <数据文件>
//     zhang_calib <数据文件> --no-refine        （跳过 LM 精化）
//
//  数据文件格式:
//     以 # 开头的是注释，自动跳过
//     第 1 行有效数据:   <视图数> <每视图点数>
//     之后每行:          X Y u v
//       X Y = 角点在标定板上的坐标（mm，Z = 0）
//       u v = 检测到的像素坐标
//     所有点按视图顺序排列
// ============================================================================
#include "zhang.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

// ---------------------------------------------------------------------------
//  打印一个 3x3 矩阵
// ---------------------------------------------------------------------------
static void printMat3(const char* name, const Mat& M) {
    std::printf("   %s = [ %12.4f  %12.4f  %12.4f ]\n", name,
                M.at(0,0), M.at(0,1), M.at(0,2));
    std::printf("   %*s   [ %12.4f  %12.4f  %12.4f ]\n", (int)std::strlen(name),
                "", M.at(1,0), M.at(1,1), M.at(1,2));
    std::printf("   %*s   [ %12.4f  %12.4f  %12.4f ]\n", (int)std::strlen(name),
                "", M.at(2,0), M.at(2,1), M.at(2,2));
}

static void printVec(const char* name, const Vec& v) {
    std::printf("   %s = (", name);
    for (std::size_t i = 0; i < v.size(); ++i)
        std::printf("%s%9.4f", i ? ", " : "", v[i]);
    std::printf(" )\n");
}

// ---------------------------------------------------------------------------
//  读取数据文件
// ---------------------------------------------------------------------------
static bool loadData(const std::string& path,
                     std::vector<View>& views,
                     std::string& err) {
    std::ifstream fin(path);
    if (!fin) { err = "无法打开文件: " + path; return false; }

    // 跳过注释和空行
    auto nextDataLine = [&](std::string& line) -> bool {
        while (std::getline(fin, line)) {
            // 去掉行首空白
            std::size_t b = line.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) continue;      // 空行
            if (line[b] == '#') continue;              // 注释
            line = line.substr(b);
            return true;
        }
        return false;
    };

    std::string line;
    if (!nextDataLine(line)) { err = "文件为空"; return false; }

    int nviews = 0, npts = 0;
    {
        std::istringstream iss(line);
        if (!(iss >> nviews >> npts)) { err = "头部格式错误，应为: 视图数 点数"; return false; }
    }
    if (nviews <= 0 || npts < 4) { err = "视图数或点数不合理"; return false; }

    views.assign(nviews, View());
    for (int i = 0; i < nviews; ++i) {
        views[i].obj.reserve(npts);
        views[i].img.reserve(npts);
        for (int j = 0; j < npts; ++j) {
            if (!nextDataLine(line)) {
                err = "数据不足，读到第 " + std::to_string(i) + " 张图第 "
                    + std::to_string(j) + " 个点时提前结束";
                return false;
            }
            double X, Y, u, v;
            std::istringstream iss(line);
            if (!(iss >> X >> Y >> u >> v)) {
                err = "解析失败: " + line;
                return false;
            }
            views[i].obj.push_back({X, Y, 0.0});
            views[i].img.push_back({u, v});
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::printf("==============================================================\n");
    std::printf("  张氏标定法 (Zhang's Camera Calibration)\n");
    std::printf("  纯 C++ 实现，不依赖 OpenCV / Eigen\n");
    std::printf("==============================================================\n");

    if (argc < 2) {
        std::printf("\n用法: %s <数据文件> [--no-refine]\n\n", argv[0]);
        std::printf("数据文件格式:\n");
        std::printf("   # 注释行\n");
        std::printf("   <视图数> <每视图点数>\n");
        std::printf("   X Y u v        (重复 视图数 * 点数 次)\n");
        return 1;
    }

    const std::string path = argv[1];
    bool doRefine = true;
    for (int i = 2; i < argc; ++i)
        if (std::strcmp(argv[i], "--no-refine") == 0) doRefine = false;

    std::vector<View> views;
    std::string err;
    if (!loadData(path, views, err)) {
        std::printf("\n[错误] %s\n", err.c_str());
        return 1;
    }

    std::printf("\n读取数据: %s\n", path.c_str());
    std::printf("  视图数: %d\n", (int)views.size());
    std::printf("  每视图点数: %d\n", (int)views[0].obj.size());
    std::printf("  总点数: %d\n", (int)views.size() * (int)views[0].obj.size());

    // -----------------------------------------------------------------------
    std::printf("\n--------------------------------------------------------------\n");
    std::printf(" 开始标定\n");
    std::printf("--------------------------------------------------------------\n");

    CalibResult res = calibrate(views, doRefine);

    if (!res.ok) {
        std::printf("\n[失败] %s\n", res.message.c_str());
        return 1;
    }

    // -----------------------------------------------------------------------
    std::printf("\n【内参矩阵 K】\n");
    printMat3("K", res.K);

    std::printf("\n【畸变系数】\n");
    std::printf("   k1 = %.10f\n", res.k1);
    std::printf("   k2 = %.10f\n", res.k2);

    std::printf("\n【外参】\n");
    for (std::size_t i = 0; i < res.R.size(); ++i) {
        std::printf("\n   第 %d 张图:\n", (int)i + 1);
        printMat3("R", res.R[i]);
        printVec("t", res.t[i]);
    }

    // -----------------------------------------------------------------------
    std::printf("\n--------------------------------------------------------------\n");
    std::printf(" 精度\n");
    std::printf("--------------------------------------------------------------\n");
    std::printf("   重投影误差 RMS = %.10f 像素\n", res.rms);
    std::printf("   %s\n", res.rms < 0.5 ? "合格（一般要求 < 0.5 像素）"
                                          : "偏大，建议检查角点提取质量");

    // -----------------------------------------------------------------------
    //  逐张图的误差，帮助定位问题
    // -----------------------------------------------------------------------
    std::printf("\n   各视图误差:\n");
    for (std::size_t i = 0; i < views.size(); ++i) {
        double sum = 0.0;
        int cnt = 0;
        for (std::size_t j = 0; j < views[i].obj.size(); ++j) {
            const double alpha = res.K.at(0,0), beta = res.K.at(1,1);
            const double gamma = res.K.at(0,1), u0 = res.K.at(0,2), v0 = res.K.at(1,2);
            // 直接算一次投影
            Vec Xc = matvec(res.R[i], { views[i].obj[j].x, views[i].obj[j].y,
                                        views[i].obj[j].z });
            Xc[0] += res.t[i][0];  Xc[1] += res.t[i][1];  Xc[2] += res.t[i][2];
            const double xn = Xc[0] / Xc[2], yn = Xc[1] / Xc[2];
            const double r2 = xn * xn + yn * yn;
            const double f = 1.0 + res.k1 * r2 + res.k2 * r2 * r2;
            const double u = alpha * (xn * f) + gamma * (yn * f) + u0;
            const double v = beta * (yn * f) + v0;
            const double du = views[i].img[j].x - u;
            const double dv = views[i].img[j].y - v;
            sum += du * du + dv * dv;
            cnt += 2;
        }
        std::printf("     第 %2d 张:  RMS = %.6f 像素\n",
                    (int)i + 1, cnt ? std::sqrt(sum / cnt) : 0.0);
    }

    std::printf("\n完成。\n");
    return 0;
}
