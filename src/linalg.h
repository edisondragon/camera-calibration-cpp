// ============================================================================
//  linalg.h  --  极简线性代数库
//
//  只依赖 C++ 标准库，不依赖 OpenCV / Eigen。
//  之所以自己写，是因为标定用到的线性代数非常有限，自己实现更容易看清每一步。
//
//  提供的功能：
//    Mat / Vec               基本矩阵向量
//    matmul, transpose       乘法、转置
//    jacobiEigen             对称矩阵特征分解（Jacobi 旋转法）
//    smallestEigenVector     求 A 的最小奇异向量（解 A*x = 0 用）
//    solveSPD                Cholesky 解对称正定方程组（LM 用）
//    det3                    3x3 行列式
//    cross3                  三维叉乘
// ============================================================================
#pragma once

#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <utility>

using Vec = std::vector<double>;

// ---------------------------------------------------------------------------
//  行主序矩阵
// ---------------------------------------------------------------------------
struct Mat {
    int rows = 0;
    int cols = 0;
    std::vector<double> a;   // 长度 rows*cols，按行存放

    Mat() = default;
    Mat(int r, int c, double v = 0.0) : rows(r), cols(c), a(std::size_t(r) * c, v) {}

    double&       at(int i, int j)       { return a[std::size_t(i) * cols + j]; }
    double        at(int i, int j) const { return a[std::size_t(i) * cols + j]; }
};

inline Mat identity(int n) {
    Mat I(n, n, 0.0);
    for (int i = 0; i < n; ++i) I.at(i, i) = 1.0;
    return I;
}

inline Mat transpose(const Mat& A) {
    Mat T(A.cols, A.rows, 0.0);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            T.at(j, i) = A.at(i, j);
    return T;
}

inline Mat matmul(const Mat& A, const Mat& B) {
    if (A.cols != B.rows) throw std::runtime_error("matmul: 维数不匹配");
    Mat C(A.rows, B.cols, 0.0);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k) {
            const double aik = A.at(i, k);
            if (aik == 0.0) continue;
            for (int j = 0; j < B.cols; ++j)
                C.at(i, j) += aik * B.at(k, j);
        }
    return C;
}

inline Vec matvec(const Mat& A, const Vec& x) {
    if ((int)x.size() != A.cols) throw std::runtime_error("matvec: 维数不匹配");
    Vec y(A.rows, 0.0);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            y[i] += A.at(i, j) * x[j];
    return y;
}

// ---------------------------------------------------------------------------
//  向量运算
// ---------------------------------------------------------------------------
inline double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline double norm(const Vec& a) { return std::sqrt(dot(a, a)); }

inline Vec normalize(const Vec& a) {
    const double n = norm(a);
    if (n < 1e-300) throw std::runtime_error("normalize: 零向量");
    Vec r = a;
    for (double& v : r) v /= n;
    return r;
}

inline Vec cross3(const Vec& a, const Vec& b) {
    return { a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0] };
}

// ---------------------------------------------------------------------------
//  3x3 行列式
// ---------------------------------------------------------------------------
inline double det3(const Mat& M) {
    return M.at(0,0) * (M.at(1,1) * M.at(2,2) - M.at(1,2) * M.at(2,1))
         - M.at(0,1) * (M.at(1,0) * M.at(2,2) - M.at(1,2) * M.at(2,0))
         + M.at(0,2) * (M.at(1,0) * M.at(2,1) - M.at(1,1) * M.at(2,0));
}

// ---------------------------------------------------------------------------
//  对称矩阵的 Jacobi 特征分解
//
//  输入:  A  (n x n，必须对称)
//  输出:  eval  特征值，按升序排列
//         V     特征向量按列存放，第 k 列对应 eval[k]
//
//  原理: 反复用平面旋转把非对角元"消零"，直到整个矩阵接近对角。
//        每消一次，特征向量矩阵就右乘一次旋转矩阵。
//        收敛后对角线上的数就是特征值。
// ---------------------------------------------------------------------------
inline void jacobiEigen(Mat A, Vec& eval, Mat& V) {
    const int n = A.rows;
    V = identity(n);

    for (int sweep = 0; sweep < 100; ++sweep) {
        // 判断非对角元的平方和是否已经足够小
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += A.at(p, q) * A.at(p, q);
        if (off < 1e-30) break;

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = A.at(p, q);
                if (std::fabs(apq) < 1e-300) continue;

                // 求旋转角 t, c, s，使得旋转后 (p,q) 位置变成 0
                const double theta = (A.at(q, q) - A.at(p, p)) / (2.0 * apq);
                const double t = ((theta >= 0.0 ? 1.0 : -1.0) /
                                  (std::fabs(theta) + std::sqrt(theta * theta + 1.0)));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                // 第一步: A <- A * J   （更新第 p, q 两列）
                for (int k = 0; k < n; ++k) {
                    const double akp = A.at(k, p), akq = A.at(k, q);
                    A.at(k, p) = c * akp - s * akq;
                    A.at(k, q) = s * akp + c * akq;
                }
                // 第二步: A <- J^T * A （更新第 p, q 两行）
                for (int k = 0; k < n; ++k) {
                    const double apk = A.at(p, k), aqk = A.at(q, k);
                    A.at(p, k) = c * apk - s * aqk;
                    A.at(q, k) = s * apk + c * aqk;
                }
                // 累积特征向量: V <- V * J
                for (int k = 0; k < n; ++k) {
                    const double vkp = V.at(k, p), vkq = V.at(k, q);
                    V.at(k, p) = c * vkp - s * vkq;
                    V.at(k, q) = s * vkp + c * vkq;
                }
            }
        }
    }

    // 收集特征值，按升序排序，同时重排特征向量
    std::vector<std::pair<double, int>> es(n);
    for (int i = 0; i < n; ++i) es[i] = { A.at(i, i), i };
    std::sort(es.begin(), es.end());

    eval.assign(n, 0.0);
    Mat Vs(n, n, 0.0);
    for (int k = 0; k < n; ++k) {
        eval[k] = es[k].first;
        for (int i = 0; i < n; ++i)
            Vs.at(i, k) = V.at(i, es[k].second);
    }
    V = Vs;
}

// ---------------------------------------------------------------------------
//  求 A 的最小奇异向量
//
//  用途: 解齐次方程 A * x = 0 的非零解。
//
//  原理: A 的最小奇异向量 = A^T * A 的最小特征值对应的特征向量。
//        （因为 A^T*A 是对称半正定的，它的特征值是 A 的奇异值的平方。）
//        这样只需要对称特征分解，不需要完整的 SVD。
// ---------------------------------------------------------------------------
inline Vec smallestEigenVector(const Mat& A) {
    Mat AtA = matmul(transpose(A), A);
    Vec eval;
    Mat V;
    jacobiEigen(AtA, eval, V);

    const int n = AtA.rows;
    Vec x(n);
    for (int i = 0; i < n; ++i) x[i] = V.at(i, 0);   // 最小特征值对应的列
    return normalize(x);
}

// ---------------------------------------------------------------------------
//  Cholesky 分解解对称正定方程组 A * x = b
//
//  原理: A = L * L^T（L 是下三角），先解 L*y = b，再解 L^T*x = y。
//  LM 算法里 J^T*J + mu*D 一定是对称正定的，所以总能解。
// ---------------------------------------------------------------------------
inline Vec solveSPD(const Mat& A, const Vec& b) {
    const int n = A.rows;
    Mat L(n, n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A.at(i, j);
            for (int k = 0; k < j; ++k) s -= L.at(i, k) * L.at(j, k);
            if (i == j) {
                if (s <= 0.0) throw std::runtime_error("solveSPD: 矩阵非正定");
                L.at(i, i) = std::sqrt(s);
            } else {
                L.at(i, j) = s / L.at(j, j);
            }
        }
    }

    Vec y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= L.at(i, k) * y[k];
        y[i] = s / L.at(i, i);
    }

    Vec x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < n; ++k) s -= L.at(k, i) * x[k];
        x[i] = s / L.at(i, i);
    }
    return x;
}

// ---------------------------------------------------------------------------
//  Rodrigues 公式: 旋转向量 -> 旋转矩阵
//  输入 w = (wx, wy, wz)，其模长就是旋转角
// ---------------------------------------------------------------------------
inline Mat rodrigues(const Vec& w) {
    const double theta = norm(w);
    Mat R = identity(3);
    if (theta < 1e-12) return R;

    const double x = w[0] / theta, y = w[1] / theta, z = w[2] / theta;
    const double c = std::cos(theta), s = std::sin(theta), C = 1.0 - c;

    R.at(0,0) = c + x*x*C;      R.at(0,1) = x*y*C - z*s;    R.at(0,2) = x*z*C + y*s;
    R.at(1,0) = y*x*C + z*s;    R.at(1,1) = c + y*y*C;      R.at(1,2) = y*z*C - x*s;
    R.at(2,0) = z*x*C - y*s;    R.at(2,1) = z*y*C + x*s;    R.at(2,2) = c + z*z*C;
    return R;
}
