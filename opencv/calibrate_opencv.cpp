// ============================================================================
//  calibrate_opencv.cpp  --  用 OpenCV 实现的同一个流程（约 130 行）
//
//  这个文件的作用是"对照"：
//    纯 C++ 版 (src/zhang.cpp) 让你看清每一步数学；
//    这个版本让你在工程中直接用，不用自己维护线性代数。
//
//  两者应该给出几乎相同的 K。如果差很多，说明你的纯 C++ 版有 bug。
//
//  编译（Linux / MinGW）:
//     g++ -O2 -std=c++17 calibrate_opencv.cpp -o calib_opencv \
//         $(pkg-config --cflags --libs opencv4)
//
//  编译（Windows + vcpkg）:
//     cl /std:c++17 /EHsc /I %VCPKG%\installed\x64-windows\include ^
//        calibrate_opencv.cpp ^
//        /link /LIBPATH:%VCPKG%\installed\x64-windows\lib opencv_world4xx.lib
//
//  用法:
//     calib_opencv <图片目录> <横排角点数> <竖排角点数> <方格边长mm>
// ============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "用法: " << argv[0]
                  << " <图片目录> <横排角点数> <竖排角点数> <方格边长mm>\n"
                  << "例如: " << argv[0] << " ./images 9 6 25\n";
        return 1;
    }

    const std::string dir   = argv[1];
    const int    cols       = std::atoi(argv[2]);   // 每行内角点数
    const int    rows       = std::atoi(argv[3]);   // 每列内角点数
    const double squareSize = std::atof(argv[4]);   // 方格实际边长（mm）

    const cv::Size patternSize(cols, rows);

    // -----------------------------------------------------------------------
    //  1. 准备标定板的物体坐标（Z = 0 平面）
    // -----------------------------------------------------------------------
    std::vector<cv::Point3f> objTemplate;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            objTemplate.emplace_back(float(c) * squareSize,
                                     float(r) * squareSize,
                                     0.0f);

    // -----------------------------------------------------------------------
    //  2. 逐张图找角点
    // -----------------------------------------------------------------------
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<fs::path> usedFiles;

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const std::string ext = e.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
            ext == ".bmp" || ext == ".JPG" || ext == ".PNG")
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cout << "[错误] 目录里没有图片: " << dir << "\n";
        return 1;
    }

    cv::Size imageSize;

    for (const auto& f : files) {
        cv::Mat img = cv::imread(f.string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "  [跳过] 读不出: " << f.filename().string() << "\n";
            continue;
        }
        imageSize = img.size();

        std::vector<cv::Point2f> corners;
        // 关键参数说明:
        //   CALIB_CB_ADAPTIVE_THRESH  自适应阈值，应对光照不均
        //   CALIB_CB_NORMALIZE_IMAGE  先做直方图均衡
        //   CALIB_CB_FAST_CHECK       快速预检，找不到就直接跳过（提速）
        bool found = cv::findChessboardCorners(
            img, patternSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH |
            cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FAST_CHECK);

        if (!found) {
            std::cout << "  [跳过] 没找到棋盘: " << f.filename().string() << "\n";
            continue;
        }

        // 亚像素精化：这一步决定了最终精度
        //   (11,11) = 搜索窗口大小，(-1,-1) = 不设死区
        //   TermCriteria 的 30 是最大迭代次数，0.001 是精度
        cv::cornerSubPix(
            img, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS +
                             cv::TermCriteria::COUNT, 30, 0.001));

        imagePoints.push_back(corners);
        objectPoints.push_back(objTemplate);
        usedFiles.push_back(f);

        std::cout << "  [成功] " << f.filename().string()
                  << "  (" << corners.size() << " 个角点)\n";
    }

    if (imagePoints.size() < 3) {
        std::cout << "[错误] 有效图片不足 3 张，无法标定\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    //  3. 标定
    // -----------------------------------------------------------------------
    cv::Mat K, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;

    // flags 里最常用的是 CALIB_FIX_K3（不需要六阶径向畸变）
    // 如果画面覆盖范围很大、畸变明显，可以去掉它
    const int flags = cv::CALIB_FIX_K3;

    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize,
                                     K, distCoeffs, rvecs, tvecs, flags);

    // -----------------------------------------------------------------------
    //  4. 输出
    // -----------------------------------------------------------------------
    std::cout << "\n================ 标定结果 ================\n";
    std::cout << "重投影误差 RMS = " << rms << " 像素\n\n";

    std::cout << "内参矩阵 K =\n" << K << "\n\n";

    std::cout << "畸变系数 (k1, k2, p1, p2, k3) =\n"
              << distCoeffs.reshape(1, 1) << "\n\n";

    // 外参：旋转向量转成矩阵更好读
    std::cout << "外参:\n";
    for (size_t i = 0; i < rvecs.size(); ++i) {
        cv::Mat R;
        cv::Rodrigues(rvecs[i], R);
        std::cout << "  第 " << (i + 1) << " 张 (" << usedFiles[i].filename().string() << ")\n";
        std::cout << "    R =\n" << R << "\n";
        std::cout << "    t = " << tvecs[i].t() << "\n";
    }

    // -----------------------------------------------------------------------
    //  5. 去畸变（对照本文档里讲过的 undistort 流程）
    // -----------------------------------------------------------------------
    // getOptimalNewCameraMatrix 的作用：
    //   去畸变会把画面"拉平"，四角会超出原图，出现黑边。
    //   这个函数重新算一个内参，让有效区域尽量填满画面。
    //   alpha = 0 -> 保留所有有效像素（会有黑边）
    //   alpha = 1 -> 全部像素都保留（边缘会有拉伸）
    cv::Mat map1, map2, newK;
    newK = cv::getOptimalNewCameraMatrix(K, distCoeffs, imageSize, 0.0, imageSize);
    cv::initUndistortRectifyMap(K, distCoeffs, cv::Mat(), newK,
                                imageSize, CV_16SC2, map1, map2);

    std::cout << "\n新的内参（去畸变后使用）=\n" << newK << "\n";

    // 处理第一张图作为演示
    {
        cv::Mat img = cv::imread(usedFiles[0].string());
        if (!img.empty()) {
            cv::Mat undistorted;
            cv::remap(img, undistorted, map1, map2, cv::INTER_LINEAR);
            const std::string out = "undistorted_sample.jpg";
            cv::imwrite(out, undistorted);
            std::cout << "已输出去畸变示例图: " << out << "\n";
        }
    }

    // -----------------------------------------------------------------------
    //  6. 保存到 YAML，方便后续程序读入
    // -----------------------------------------------------------------------
    {
        cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE);
        fs << "image_width"  << imageSize.width;
        fs << "image_height" << imageSize.height;
        fs << "camera_matrix" << K;
        fs << "distortion_coefficients" << distCoeffs;
        fs << "new_camera_matrix" << newK;
        fs << "rms" << rms;
        fs.release();
        std::cout << "已保存标定参数: calibration.yml\n";
    }

    return 0;
}
