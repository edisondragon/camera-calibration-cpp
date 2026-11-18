# cpp-zhang-calib —— 张氏标定法纯 C++ 实现

一个**不依赖 OpenCV / Eigen** 的张氏标定法完整实现。

代码结构严格对应 [张氏标定法-完整公式与推导.md](docs/张氏标定法-完整公式与推导.md) 的章节，
每个函数都在注释里标了对应的章节号，可以对照阅读。

---

## 为什么不用 OpenCV

用 OpenCV 三行就能标定完：

```cpp
cv::calibrateCamera(objectPoints, imagePoints, imageSize, K, dist, rvecs, tvecs);
```

但你看不到里面发生了什么。这个项目的目的是**让每一步数学都暴露在代码里**：

| 文档章节 | 内容 | 代码位置 |
|---|---|---|
| 第 1 章 | 相机模型、内参矩阵的逆 | `KfromB` 里的 Kinv 构造、`extrinsicsFromH` |
| 第 3 章 | DLT + Hartley 归一化求单应 | `computeHomography` |
| 第 5 章 | B 矩阵的六个元素 | `KfromB` 内部的六个公式 |
| 第 6 章 | v(ij) 的构造、V*b = 0 | `vij`、`calibrate` 步骤 2~3 |
| 第 7 章 | 从 B 反解内参（逐步推导） | `KfromB` |
| 第 8 章 | 外参 + SVD 正交化 | `extrinsicsFromH`、`nearestRotation` |
| 第 9 章 | 径向畸变线性最小二乘 | `estimateDistortion` |
| 第 10 章 | LM 精化 + 雅可比四个 Block | `refineLM`、`calcJacobian` |

---

## 目录结构

```
cpp-zhang-calib/
├── CMakeLists.txt
├── README.md                    本文件
├── src/
│   ├── linalg.h                 极简线性代数（矩阵、Jacobi 特征分解、Cholesky）
│   ├── zhang.h                  接口定义
│   ├── zhang.cpp                标定算法主体
│   └── main.cpp                 命令行入口
├── opencv/
│   └── calibrate_opencv.cpp     OpenCV 版对照实现（含角点检测）
├── tools/
│   └── zhang_reference.py       Python 参考实现（算法验证 + 生成测试数据）
└── data/
    └── synthetic.txt            合成测试数据（5 视图 × 49 点）
```

---

## 编译

### 任意平台（CMake）

```bash
cd cpp-zhang-calib
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

生成的可执行文件在 `bin/zhang_calib`（Windows 下同一路径，加 `.exe` 后缀）。

### 直接编译（不用 CMake）

```bash
# Linux / macOS / MinGW
g++ -O2 -std=c++17 -Isrc src/main.cpp src/zhang.cpp -o zhang_calib

# MSVC
cl /std:c++17 /EHsc /O2 /Isrc src\main.cpp src\zhang.cpp /Fe:zhang_calib.exe
```

**要求**：支持 C++17 的编译器。没有其他依赖。

---

## 运行

```bash
./zhang_calib data/synthetic.txt
```

### 数据文件格式

```
# 以 # 开头的是注释，自动跳过
5 49                          <-- 视图数  每视图点数
0.0   0.0   815.3  297.1      <-- X   Y   u   v
30.0  0.0   851.6  299.4
...
```

- `X Y` 是角点在标定板上的坐标（mm，板平面为 Z = 0）
- `u v` 是检测到的像素坐标
- 所有点按视图顺序排列（视图 1 的所有点，然后视图 2……）

### 选项

```bash
./zhang_calib data/synthetic.txt --no-refine    # 跳过 LM 精化，只看线性闭式解
```

---

## 验证结果

项目自带 Python 参考实现 `tools/zhang_reference.py`，它是 `zhang.cpp` 的逐行镜像。
运行它可以确认算法正确：

```bash
python tools/zhang_reference.py
```

实测输出（无噪声合成数据）：

```
[验证 0] Jacobi 特征分解 vs numpy.linalg.eigh
   最大绝对误差     : 4.441e-15
   特征方程残差     : 4.441e-15
   --> 通过

[验证 2] 线性闭式解
   真值 K =             估计 K =
   [[1200.  0.  640.]   [[1200.  0.  640.]
    [   0. 1150. 480.]    [   0. 1150. 480.]
    [   0.    0.   1.]]   [   0.    0.   1.]]

   绝对误差:
     alpha: 真值  1200.000000   估计  1200.000000   误差 2.274e-15
     beta : 真值  1150.000000   估计  1150.000000   误差 2.373e-15
     u0   : 真值   640.000000   估计   640.000000   误差 3.908e-15
     v0   : 真值   480.000000   估计   480.000000   误差 1.184e-15

[验证 3] 外参正交性
   第 1 张: max|R^T R - I| = 4.219e-16 ,  det(R) = 1.000000
   ...

[验证 4] 重投影误差
   重投影误差 RMS = 1.958e-13 像素

 结论: 算法正确 —— 内参完全复原，重投影误差为机器精度
```

**结论**：内参恢复到机器精度，说明 DLT、Hartley 归一化、V*b = 0、
B 反解内参、外参正交化这几步的实现都是正确的。

---

## 算法实现中的几个关键细节

这些是"照着论文写会出错"的地方，代码里都处理了。

### 1. b 的符号必须修正

`smallestEigenVector` 返回的是单位向量，符号是任意的。
如果 b 整体是负的，那么 B 也是负定矩阵，
第 7 章的 `lambda = B33 - ...` 会算出负数，开根号直接变成 NaN。

```cpp
// 符号修正: B 必须正定，所以 B11 > 0
if (b[0] < 0.0) for (double& v : b) v = -v;
```

### 2. H 的符号歧义

`H` 和 `-H` 给出**完全相同的投影**，但解出的位姿不同：

```
   H  ->  ( R,              t  )
  -H  ->  ( R*diag(-1,-1,1), -t )     <- 投影一样，但板跑到相机后面了
```

如果不管，约有一半的视图会解出负深度。代码里统一选择"板在相机前方"：

```cpp
if (t[2] < 0.0) {
    R.at(i,0) = -R.at(i,0);   // R <- R * diag(-1,-1,1)
    R.at(i,1) = -R.at(i,1);
    for (double& v : t) v = -v;
}
```

### 3. 正交化必须处理 det = -1

把 `Q = [r1 r2 r3]` 投影到最近的旋转矩阵，用的是

```
   R = Q * (Q^T Q)^(-1/2)
```

但如果 `det(Q) < 0`，这样得到的 R 行列式是 -1（是镜像不是旋转）。
代码里在求 `(Q^T Q)^(-1/2)` 时对最后一个特征方向翻符号来修正。

### 4. 只有 2 张图时需要补约束

b 有 6 个未知数、去掉整体尺度后是 5 个。
2 张图只能给 4 个方程，欠定 1 个，需要补 `skew = 0` 即 `B12 = 0`：

```cpp
if (nv == 2) V.at(4, 1) = 1.0;   // b[1] = B12 = 0
```

---

## 局限性

**这个项目不含角点检测。** 角点检测（Harris、Shi-Tomasi、亚像素精化）
本身是另一个大话题，和标定算法是两件事。

所以它**不能直接吃照片**。要处理真实照片，有三条路：

### 路线 A：用 OpenCV 版本（推荐）

`opencv/calibrate_opencv.cpp` 包含了完整的角点检测 + 标定 + 去畸变：

```bash
g++ -O2 -std=c++17 opencv/calibrate_opencv.cpp -o calib_opencv \
    $(pkg-config --cflags --libs opencv4)
./calib_opencv ./images 9 6 25
```

参数依次是：图片目录、横排内角点数、竖排内角点数、方格边长（mm）。

### 路线 B：用 OpenCV 提角点，导出文本，喂给本项目

这样你可以用真实数据检验这份实现：

```python
import cv2, numpy as np
# ... 用 findChessboardCorners + cornerSubPix 提角点 ...
# 按 data/synthetic.txt 的格式写成文本
```

### 路线 C：直接从别的标定工具导出角点

MATLAB 标定工具箱、`camera_calibration` ROS 包等都能导出角点坐标。

---

## 想进一步改？

几个自然的练习方向：

1. **加切向畸变 p1, p2**
   在 `estimateDistortion` 里，每点从 2 个未知数扩到 4 个，矩阵变得一样简单。

2. **给 LM 加 Schur 补**
   现在的 `refineLM` 用稠密法方程解，参数量是 `7 + 6n`。
   10 张图是 67 个参数，还能算；100 张图就慢了。
   标准做法是利用"每张图的外参只出现在自己的残差里"这个块对角结构做 Schur 补，
   把问题降成只解 7 个内参。

3. **加更多畸变模型**
   鱼眼要用等距投影模型，不是多项式。

4. **加鲁棒核函数**
   现在是最小二乘，一个错误角点会污染整个解。
   换成 Huber 或 Cauchy 核能让标定对离群点更稳。

5. **加自动判断标定质量**
   比如检查各视图的覆盖范围、姿态多样性，给出"数据够不够好"的建议。
