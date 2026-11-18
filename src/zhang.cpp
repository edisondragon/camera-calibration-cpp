// ============================================================================
//  zhang.cpp  --  张氏标定法完整实现
//
//  代码结构和《张氏标定法-完整公式与推导.md》的章节一一对应，
//  每个函数开头都标注了对应的章节号，方便对照阅读。
// ============================================================================
#include "zhang.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <algorithm>

// ===========================================================================
//  内部小工具
// ===========================================================================
namespace {

// 取 3x3 矩阵的第 c 列（c = 0,1,2），返回 3 维向量
Vec col3(const Mat& M, int c) {
    return { M.at(0, c), M.at(1, c), M.at(2, c) };
}

// 构造 3x3 齐次平移缩放矩阵:  [ s 0 -s*cx ]
//                             [ 0 s -s*cy ]
//                             [ 0 0   1   ]
Mat makeNormalizer(double cx, double cy, double s) {
    Mat T(3, 3, 0.0);
    T.at(0,0) = s;  T.at(1,1) = s;
    T.at(0,2) = -s * cx;  T.at(1,2) = -s * cy;
    T.at(2,2) = 1.0;
    return T;
}

// 求归一化参数：重心 (cx,cy) 和缩放系数 s（使平均距离 = sqrt(2)）
// 对应文档 3.6 节
void normalizationParams(const std::vector<Point2>& pts,
                         double& cx, double& cy, double& s) {
    const int n = (int)pts.size();
    cx = 0.0; cy = 0.0;
    for (const auto& p : pts) { cx += p.x; cy += p.y; }
    cx /= n; cy /= n;

    double d = 0.0;
    for (const auto& p : pts) d += std::hypot(p.x - cx, p.y - cy);
    d /= n;
    s = std::sqrt(2.0) / d;
}

// 对应文档 6.2 节的 v(ij)
//   v(ij) = ( h_i1*h_j1,  h_i1*h_j2 + h_i2*h_j1,  h_i2*h_j2,
//             h_i1*h_j3 + h_i3*h_j1,  h_i2*h_j3 + h_i3*h_j2,  h_i3*h_j3 )
Vec vij(const Vec& hi, const Vec& hj) {
    return { hi[0] * hj[0],
             hi[0] * hj[1] + hi[1] * hj[0],
             hi[1] * hj[1],
             hi[0] * hj[2] + hi[2] * hj[0],
             hi[1] * hj[2] + hi[2] * hj[1],
             hi[2] * hj[2] };
}

// 把 Q 投影到最近的合法旋转矩阵（正交 Procrustes）
// 对应文档 8.3 节
Mat nearestRotation(const Mat& Q) {
    // R = Q * (Q^T Q)^(-1/2)
    // 令 Q^T Q = V * L * V^T（对称正定），则 (Q^T Q)^(-1/2) = V * L^(-1/2) * V^T
    Mat QtQ = matmul(transpose(Q), Q);
    Vec lam;
    Mat V;
    jacobiEigen(QtQ, lam, V);

    for (int i = 0; i < 3; ++i)
        if (lam[i] <= 1e-300) throw std::runtime_error("nearestRotation: Q 退化");

    // 判断是否需要翻转符号（使 det(R) = +1）
    const double detQ = det3(Q);
    double flip[3] = { 1.0, 1.0, 1.0 };
    if (detQ < 0.0) flip[2] = -1.0;

    Mat W(3, 3, 0.0);       // W = V * diag(flip / sqrt(lam)) * V^T
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k)
                acc += V.at(i, k) * (flip[k] / std::sqrt(lam[k])) * V.at(j, k);
            W.at(i, j) = acc;
        }
    return matmul(Q, W);
}

// 投影一个点，返回所有中间量（精化阶段反复用到）
struct ProjInfo {
    Vec Xc;              // 相机坐标
    double xn = 0, yn = 0;
    double r2 = 0;
    double f  = 0;
    double xd = 0, yd = 0;
    double u  = 0, v  = 0;
};

ProjInfo project(const Mat& R, const Vec& t, const Point3& M,
                 double alpha, double beta, double gamma,
                 double u0, double v0, double k1, double k2) {
    ProjInfo P;
    P.Xc = matvec(R, { M.x, M.y, M.z });
    P.Xc[0] += t[0];  P.Xc[1] += t[1];  P.Xc[2] += t[2];

    P.xn = P.Xc[0] / P.Xc[2];
    P.yn = P.Xc[1] / P.Xc[2];
    P.r2 = P.xn * P.xn + P.yn * P.yn;
    P.f  = 1.0 + k1 * P.r2 + k2 * P.r2 * P.r2;
    P.xd = P.xn * P.f;
    P.yd = P.yn * P.f;
    P.u  = alpha * P.xd + gamma * P.yd + u0;
    P.v  = beta  * P.yd + v0;
    return P;
}

} // namespace


// ===========================================================================
//  1. 单应矩阵（DLT + Hartley 归一化）        -- 对应文档第 3 章
// ===========================================================================
Mat computeHomography(const std::vector<Point3>& obj,
                      const std::vector<Point2>& img) {
    const int n = (int)obj.size();
    if (n < 4) throw std::runtime_error("computeHomography: 至少需要 4 个点");
    if ((int)img.size() != n) throw std::runtime_error("computeHomography: 点数不一致");

    // ---- 1a. 归一化参数 ----
    std::vector<Point2> obj2(n);
    for (int i = 0; i < n; ++i) obj2[i] = { obj[i].x, obj[i].y };

    double cu, cv, su;      normalizationParams(img,  cu, cv, su);
    double cX, cY, sX;      normalizationParams(obj2, cX, cY, sX);

    // ---- 1b. 构造 A（2n 行 9 列）---- 文档 3.3 / 3.4 节
    Mat A(2 * n, 9, 0.0);
    for (int i = 0; i < n; ++i) {
        const double X = sX * (obj[i].x - cX);   // 归一化后的板坐标
        const double Y = sX * (obj[i].y - cY);
        const double u = su * (img[i].x - cu);   // 归一化后的像素
        const double v = su * (img[i].y - cv);

        double* r0 = &A.a[std::size_t(2 * i) * 9];
        double* r1 = &A.a[std::size_t(2 * i + 1) * 9];

        // 第 1 个分量 v*C - B = 0  ->  [ 0 0 0 -X -Y -1  vX vY v ]
        r0[3] = -X;  r0[4] = -Y;  r0[5] = -1.0;
        r0[6] =  v * X;  r0[7] = v * Y;  r0[8] = v;

        // 第 2 个分量 A - u*C = 0  ->  [ X Y 1 0 0 0 -uX -uY -u ]
        r1[0] = X;  r1[1] = Y;  r1[2] = 1.0;
        r1[6] = -u * X;  r1[7] = -u * Y;  r1[8] = -u;
    }

    // ---- 1c. 最小奇异向量 ---- 文档 3.5 节
    Vec h = smallestEigenVector(A);

    Mat Hn(3, 3, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Hn.at(i, j) = h[i * 3 + j];

    // ---- 1d. 还原 H = T^(-1) * Hn * S ---- 文档 3.6 节
    Mat Tinv(3, 3, 0.0);
    Tinv.at(0,0) = 1.0 / su;  Tinv.at(1,1) = 1.0 / su;
    Tinv.at(0,2) = cu;        Tinv.at(1,2) = cv;
    Tinv.at(2,2) = 1.0;

    Mat S = makeNormalizer(cX, cY, sX);

    Mat H = matmul(matmul(Tinv, Hn), S);

    // 统一尺度：让 H(2,2) = 1（便于阅读；对投影结果无影响）
    const double h22 = H.at(2, 2);
    if (std::fabs(h22) > 1e-300)
        for (double& v : H.a) v /= h22;

    return H;
}


// ===========================================================================
//  2. 从 b 解出内参 K                          -- 对应文档第 7 章
// ===========================================================================
Mat KfromB(const Vec& b) {
    // b = ( B11, B12, B22, B13, B23, B33 )
    const double B11 = b[0], B12 = b[1], B22 = b[2];
    const double B13 = b[3], B23 = b[4], B33 = b[5];

    const double denom = B11 * B22 - B12 * B12;
    if (std::fabs(denom) < 1e-300)
        throw std::runtime_error("KfromB: B11*B22 - B12^2 接近 0，数据退化");

    // 文档 7.3 节
    const double v0 = (B12 * B13 - B11 * B23) / denom;

    // 文档 7.4 节
    const double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    if (lambda <= 0.0)
        throw std::runtime_error("KfromB: lambda <= 0，说明 b 的符号或数据有问题");

    // 文档 7.5 ~ 7.8 节
    const double alpha = std::sqrt(lambda / B11);
    const double beta  = std::sqrt(lambda * B11 / denom);
    const double gamma = -B12 * alpha * alpha * beta / lambda;
    const double u0    = gamma * v0 / beta - B13 * alpha * alpha / lambda;

    Mat K(3, 3, 0.0);
    K.at(0,0) = alpha;  K.at(0,1) = gamma;  K.at(0,2) = u0;
    K.at(1,1) = beta;                        K.at(1,2) = v0;
    K.at(2,2) = 1.0;
    return K;
}


// ===========================================================================
//  3. 从 H 和 K 求外参 R, t                    -- 对应文档第 8 章
// ===========================================================================
void extrinsicsFromH(const Mat& H, const Mat& K, Mat& R, Vec& t) {
    // K 的逆（用文档 1.4 节的闭式公式，不必真的求逆）
    const double alpha = K.at(0,0), beta = K.at(1,1);
    const double gamma = K.at(0,1), u0 = K.at(0,2), v0 = K.at(1,2);

    Mat Kinv(3, 3, 0.0);
    Kinv.at(0,0) = 1.0 / alpha;
    Kinv.at(0,1) = -gamma / (alpha * beta);
    Kinv.at(0,2) = (gamma * v0 - beta * u0) / (alpha * beta);
    Kinv.at(1,1) = 1.0 / beta;
    Kinv.at(1,2) = -v0 / beta;
    Kinv.at(2,2) = 1.0;

    const Vec h1 = col3(H, 0), h2 = col3(H, 1), h3 = col3(H, 2);

    const Vec a1 = matvec(Kinv, h1);
    const Vec a2 = matvec(Kinv, h2);
    const Vec a3 = matvec(Kinv, h3);

    // 文档 8.1 节：lambda 由 ||r1|| = ||r2|| = 1 定出
    const double n1 = norm(a1), n2 = norm(a2);
    const double lambda = (n1 > 1e-300 && n2 > 1e-300)
                        ? 2.0 / (n1 + n2)
                        : 1.0 / n1;

    Vec r1 = a1, r2 = a2, tt = a3;
    for (double& v : r1) v *= lambda;
    for (double& v : r2) v *= lambda;
    for (double& v : tt) v *= lambda;

    // r1, r2 归一化到单位长度，减少噪声影响
    r1 = normalize(r1);
    r2 = normalize(r2);

    Vec r3 = cross3(r1, r2);

    // 文档 8.3 节：拼成 Q，投影到最近的合法旋转矩阵
    Mat Q(3, 3, 0.0);
    for (int i = 0; i < 3; ++i) {
        Q.at(i,0) = r1[i];  Q.at(i,1) = r2[i];  Q.at(i,2) = r3[i];
    }
    R = nearestRotation(Q);

    // 消除 H 整体符号带来的歧义：
    // H 和 -H 给出相同投影，但一个让标定板在相机前方、另一个在后方。
    // 这里统一选择"板在相机前方"（相机坐标 Z > 0）的那个解。
    t = tt;
    if (t[2] < 0.0) {
        // 等价变换：R <- R * diag(-1,-1,1), t <- -t，投影不变
        for (int i = 0; i < 3; ++i) {
            R.at(i,0) = -R.at(i,0);
            R.at(i,1) = -R.at(i,1);
        }
        for (double& v : t) v = -v;
    }
}


// ===========================================================================
//  4. 径向畸变（线性最小二乘）                 -- 对应文档第 9 章
// ===========================================================================
void estimateDistortion(const std::vector<View>& views,
                        const std::vector<Mat>& H,
                        const Mat& K,
                        double& k1, double& k2) {
    k1 = 0.0; k2 = 0.0;

    const double alpha = K.at(0,0), beta = K.at(1,1);
    const double gamma = K.at(0,1), u0 = K.at(0,2), v0 = K.at(1,2);

    Mat Kinv(3, 3, 0.0);
    Kinv.at(0,0) = 1.0 / alpha;
    Kinv.at(0,1) = -gamma / (alpha * beta);
    Kinv.at(0,2) = (gamma * v0 - beta * u0) / (alpha * beta);
    Kinv.at(1,1) = 1.0 / beta;
    Kinv.at(1,2) = -v0 / beta;
    Kinv.at(2,2) = 1.0;

    std::vector<double> Drows, drows;

    for (std::size_t i = 0; i < views.size(); ++i) {
        const Mat KH = matmul(Kinv, H[i]);
        for (std::size_t j = 0; j < views[i].obj.size(); ++j) {
            const Point3& M = views[i].obj[j];
            const Point2& p = views[i].img[j];

            // 理想归一化坐标：由 K^(-1) * H * (X,Y,1) 得到
            Vec q = matvec(KH, { M.x, M.y, 1.0 });
            const double xn = q[0] / q[2];
            const double yn = q[1] / q[2];

            // 观测归一化坐标：由像素反算
            const double yd = (p.y - v0) / beta;
            const double xd = (p.x - u0 - gamma * yd) / alpha;

            const double r2 = xn * xn + yn * yn;

            // 每点两行:  [ xn*r2  xn*r4 ] [k1]   [ xd - xn ]
            //            [ yn*r2  yn*r4 ] [k2] = [ yd - yn ]
            Drows.push_back(xn * r2);  Drows.push_back(xn * r2 * r2);
            drows.push_back(xd - xn);

            Drows.push_back(yn * r2);  Drows.push_back(yn * r2 * r2);
            drows.push_back(yd - yn);
        }
    }

    const int m = (int)drows.size() / 2;
    if (m < 3) return;

    // 最小二乘: k = (D^T D)^(-1) D^T d
    Mat DD(2, 2, 0.0);
    Vec Dd(2, 0.0);
    for (int i = 0; i < m; ++i) {
        const double a0 = Drows[2*i], a1 = Drows[2*i + 1];
        const double b0 = drows[2*i], b1 = drows[2*i + 1];
        DD.at(0,0) += a0 * a0;  DD.at(0,1) += a0 * a1;
        DD.at(1,0) += a1 * a0;  DD.at(1,1) += a1 * a1;
        Dd[0] += a0 * b0;       Dd[1] += a1 * b1;
    }

    try {
        Vec k = solveSPD(DD, Dd);
        k1 = k[0];  k2 = k[1];
    } catch (...) {
        k1 = 0.0;  k2 = 0.0;
    }
}


// ===========================================================================
//  5. 重投影误差 RMS
// ===========================================================================
double reprojectionRMS(const std::vector<View>& views, const CalibResult& res) {
    const double alpha = res.K.at(0,0), beta = res.K.at(1,1);
    const double gamma = res.K.at(0,1), u0 = res.K.at(0,2), v0 = res.K.at(1,2);

    double sum = 0.0;
    int cnt = 0;
    for (std::size_t i = 0; i < views.size(); ++i) {
        for (std::size_t j = 0; j < views[i].obj.size(); ++j) {
            ProjInfo P = project(res.R[i], res.t[i], views[i].obj[j],
                                 alpha, beta, gamma, u0, v0, res.k1, res.k2);
            const double du = views[i].img[j].x - P.u;
            const double dv = views[i].img[j].y - P.v;
            sum += du * du + dv * dv;
            cnt += 2;
        }
    }
    return cnt > 0 ? std::sqrt(sum / cnt) : 0.0;
}


// ===========================================================================
//  6. Levenberg-Marquardt 精化                  -- 对应文档第 10 章
// ===========================================================================
void refineLM(const std::vector<View>& views, CalibResult& res) {
    const int nv = (int)views.size();

    // ---- 当前估计 ----
    double alpha = res.K.at(0,0), gamma = res.K.at(0,1), u0 = res.K.at(0,2);
    double beta  = res.K.at(1,1), v0    = res.K.at(1,2);
    double k1 = res.k1, k2 = res.k2;
    std::vector<Mat> R = res.R;
    std::vector<Vec> t = res.t;

    // ---- 参数向量布局 ----
    //   [0..6]  : d_alpha, d_beta, d_gamma, d_u0, d_v0, d_k1, d_k2
    //   之后每张图 6 个: 旋转增量 dw(3) + 平移增量 dt(3)
    const int NP = 7 + 6 * nv;

    int nres = 0;
    for (const auto& vw : views) nres += 2 * (int)vw.obj.size();

    // 计算残差向量（观测 - 预测）
    auto calcResidual = [&](Vec& r) {
        r.assign(nres, 0.0);
        int idx = 0;
        for (int i = 0; i < nv; ++i) {
            for (std::size_t j = 0; j < views[i].obj.size(); ++j) {
                ProjInfo P = project(R[i], t[i], views[i].obj[j],
                                     alpha, beta, gamma, u0, v0, k1, k2);
                r[idx++] = views[i].img[j].x - P.u;
                r[idx++] = views[i].img[j].y - P.v;
            }
        }
    };

    auto cost = [&](const Vec& r) {
        double s = 0.0;
        for (double v : r) s += v * v;
        return 0.5 * s;
    };

    // 计算雅可比矩阵
    auto calcJacobian = [&](Mat& J) {
        J = Mat(nres, NP, 0.0);
        int row = 0;
        for (int i = 0; i < nv; ++i) {
            const int base = 7 + 6 * i;
            for (std::size_t j = 0; j < views[i].obj.size(); ++j) {
                ProjInfo P = project(R[i], t[i], views[i].obj[j],
                                     alpha, beta, gamma, u0, v0, k1, k2);
                const Vec& Xc = P.Xc;
                const double Zc = Xc[2];
                const double invZ = 1.0 / Zc;

                // ---- Block 2: (xd,yd) 对 (xn,yn) ----
                const double d1 = P.f + 2.0 * k1 * P.xn * P.xn
                                       + 4.0 * k2 * P.xn * P.xn * P.r2;
                const double d2 = 2.0 * k1 * P.xn * P.yn
                                       + 4.0 * k2 * P.xn * P.yn * P.r2;
                const double d3 = P.f + 2.0 * k1 * P.yn * P.yn
                                       + 4.0 * k2 * P.yn * P.yn * P.r2;

                // ---- Block 3: (xn,yn) 对 (Xc,Yc,Zc) ----
                //  d(xn,yn)/d(Xc,Yc,Zc) = [ 1/Zc   0    -Xc/Zc^2 ]
                //                         [  0   1/Zc   -Yc/Zc^2 ]
                //
                // ---- Block 1 与 2、3 合并: (u,v) 对 (Xc,Yc,Zc) ----
                // [alpha gamma] * B2 * B3
                // [  0   beta ]
                double M2[2][3];
                const double B2[2][2] = { { d1, d2 }, { d2, d3 } };
                const double B3[2][3] = { { invZ, 0.0, -Xc[0] * invZ * invZ },
                                          { 0.0, invZ, -Xc[1] * invZ * invZ } };
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 3; ++b)
                        M2[a][b] = B2[a][0] * B3[0][b] + B2[a][1] * B3[1][b];

                // 再乘 Block 1（内参）
                double M3[2][3];
                for (int b = 0; b < 3; ++b) {
                    M3[0][b] = alpha * M2[0][b] + gamma * M2[1][b];
                    M3[1][b] = beta  * M2[1][b];
                }

                // ---- Block 4: (Xc,Yc,Zc) 对旋转增量 dw ----
                //  dXc/dw = -[Xc]x,  其中 [Xc]x = [ 0  -Zc   Yc ]
                //                                 [ Zc   0  -Xc ]
                //                                 [-Yc   Xc   0  ]
                // 于是 dXc/dw 的三列分别是:
                //    col0 = -[Xc]x * e0 = -(0, Zc, -Yc)  = (0, -Zc, Yc)
                //    col1 = -( -Zc, 0, Xc) = (Zc, 0, -Xc)
                //    col2 = -( Yc, -Xc, 0) = (-Yc, Xc, 0)
                const double dXdw[3][3] = {
                    {  0.0,   Zc,  -Xc[1] },
                    { -Zc,   0.0,   Xc[0] },
                    {  Xc[1], -Xc[0], 0.0 }
                };

                // J_rot = M3 * dXdw       (2x3)
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 3; ++b) {
                        double acc = 0.0;
                        for (int c = 0; c < 3; ++c)
                            acc += M3[a][c] * dXdw[c][b];
                        J.at(row + a, base + b) = -acc;    // 残差 = 观测 - 预测，故取负
                    }

                // J_t = M3 * I = M3       (2x3)
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 3; ++b)
                        J.at(row + a, base + 3 + b) = -M3[a][b];

                // ---- 对内参 (alpha, beta, gamma, u0, v0) ----
                const double dintr[2][5] = {
                    { P.xd, 0.0,   P.yd, 1.0, 0.0 },
                    { 0.0,  P.yd,  0.0,  0.0, 1.0 }
                };
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 5; ++b)
                        J.at(row + a, b) = -dintr[a][b];

                // ---- 对畸变 (k1, k2) ----
                J.at(row + 0, 5) = -(alpha * P.xn * P.r2 + gamma * P.yn * P.r2);
                J.at(row + 0, 6) = -(alpha * P.xn * P.r2 * P.r2 + gamma * P.yn * P.r2 * P.r2);
                J.at(row + 1, 5) = -(beta * P.yn * P.r2);
                J.at(row + 1, 6) = -(beta * P.yn * P.r2 * P.r2);

                row += 2;
            }
        }
    };

    // ---- LM 主循环 ----
    Vec r;
    calcResidual(r);
    double F = cost(r);

    double mu = 1e-3;
    double nu = 2.0;

    const int maxIter = 60;
    for (int iter = 0; iter < maxIter; ++iter) {
        Mat J;
        calcJacobian(J);

        // 梯度 g = J^T * r
        Vec g = matvec(transpose(J), r);
        const double ginf = *std::max_element(g.begin(), g.end(),
            [](double a, double b) { return std::fabs(a) < std::fabs(b); });
        if (std::fabs(ginf) < 1e-12) break;

        // 法方程矩阵 A = J^T J
        Mat A = matmul(transpose(J), J);

        // 阻尼矩阵 D = diag(A)，并按标准做法把 mu 初值定成 tau * max(diag)
        Vec D(NP, 0.0);
        double maxDiag = 0.0;
        for (int i = 0; i < NP; ++i) {
            D[i] = A.at(i, i);
            maxDiag = std::max(maxDiag, D[i]);
        }
        if (iter == 0) mu = 1e-3 * (maxDiag > 0 ? maxDiag : 1.0);

        // 解 (A + mu*D) * delta = -g
        bool stepped = false;
        for (int tries = 0; tries < 30; ++tries) {
            Mat Ad = A;
            for (int i = 0; i < NP; ++i)
                Ad.at(i, i) += mu * std::max(D[i], 1e-12);

            Vec delta;
            try {
                delta = solveSPD(Ad, g);
                for (double& v : delta) v = -v;
            } catch (...) {
                mu *= nu;  nu *= 2.0;  continue;
            }

            // 预测下降量  dL = 0.5 * delta^T * (mu*D*delta - g)
            double dL = 0.0;
            for (int i = 0; i < NP; ++i)
                dL += delta[i] * (mu * std::max(D[i], 1e-12) * delta[i] - g[i]);
            dL *= 0.5;
            if (dL <= 0.0) { mu *= nu;  nu *= 2.0;  continue; }

            // ---- 试走一步：把增量作用到参数上 ----
            const double OA = alpha, OB = beta, OG = gamma, OU = u0, OV = v0;
            const double OK1 = k1, OK2 = k2;
            std::vector<Mat> OR = R;
            std::vector<Vec> OT = t;

            alpha += delta[0];  beta  += delta[1];  gamma += delta[2];
            u0    += delta[3];  v0    += delta[4];
            k1    += delta[5];  k2    += delta[6];

            for (int i = 0; i < nv; ++i) {
                const int base = 7 + 6 * i;
                Vec dw = { delta[base], delta[base + 1], delta[base + 2] };
                Vec dt = { delta[base + 3], delta[base + 4], delta[base + 5] };
                // 旋转增量左乘（文档 10.2 节 Block 4 的推导依据）
                R[i] = matmul(rodrigues(dw), R[i]);
                for (int k = 0; k < 3; ++k) t[i][k] += dt[k];
            }

            Vec rNew;
            calcResidual(rNew);
            const double Fnew = cost(rNew);
            const double rho = (F - Fnew) / dL;

            if (rho > 0.0) {
                // 接受
                r = rNew;
                F = Fnew;
                mu *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
                nu = 2.0;
                stepped = true;

                if (norm(delta) < 1e-12) { iter = maxIter; }
                break;
            } else {
                // 拒绝，回滚
                alpha = OA;  beta = OB;  gamma = OG;  u0 = OU;  v0 = OV;
                k1 = OK1;    k2 = OK2;
                R = OR;      t = OT;
                mu *= nu;    nu *= 2.0;
            }
        }
        if (!stepped) {
            // 连续 30 次尝试都被拒绝，认为已经无法改进
            if (mu > 1e12) break;
        }
    }

    // ---- 写回 ----
    res.K.at(0,0) = alpha;  res.K.at(0,1) = gamma;  res.K.at(0,2) = u0;
    res.K.at(1,1) = beta;                            res.K.at(1,2) = v0;
    res.k1 = k1;  res.k2 = k2;
    res.R = R;    res.t = t;
}


// ===========================================================================
//  7. 顶层流程
// ===========================================================================
CalibResult calibrate(const std::vector<View>& views, bool doRefine) {
    CalibResult res;

    const int nv = (int)views.size();
    if (nv < 3) {
        res.message = "至少需要 3 张图（2 张图需要额外假设 skew = 0）";
        return res;
    }

    try {
        // ---- 步骤 1: 每张图的单应矩阵 ----
        std::vector<Mat> H(nv);
        for (int i = 0; i < nv; ++i)
            H[i] = computeHomography(views[i].obj, views[i].img);

        // ---- 步骤 2: 构造 V（每张图两行）----
        int Vrows = 2 * nv + (nv == 2 ? 1 : 0);
        Mat V(Vrows, 6, 0.0);
        for (int i = 0; i < nv; ++i) {
            const Vec h1 = col3(H[i], 0);
            const Vec h2 = col3(H[i], 1);
            const Vec v12 = vij(h1, h2);
            const Vec v11 = vij(h1, h1);
            const Vec v22 = vij(h2, h2);
            for (int k = 0; k < 6; ++k) {
                V.at(2 * i,     k) = v12[k];
                V.at(2 * i + 1, k) = v11[k] - v22[k];
            }
        }
        // 只有 2 张图时补充 skew = 0 约束: B12 = 0
        if (nv == 2) V.at(4, 1) = 1.0;

        // ---- 步骤 3: 解 V * b = 0 ----
        Vec b = smallestEigenVector(V);

        // 符号修正: B 必须正定，所以 B11 > 0。
        // SVD 给出的符号是任意的，这里统一成正号，否则 lambda 会变成负数。
        if (b[0] < 0.0) for (double& v : b) v = -v;

        // ---- 步骤 4: 从 b 解内参 K ----
        res.K = KfromB(b);

        // 合理性检查
        if (!(res.K.at(0,0) > 0) || !(res.K.at(1,1) > 0))
            throw std::runtime_error("解出的焦距不是正数，数据可能退化");

        // ---- 步骤 5: 外参 ----
        res.R.resize(nv);
        res.t.resize(nv);
        for (int i = 0; i < nv; ++i)
            extrinsicsFromH(H[i], res.K, res.R[i], res.t[i]);

        // ---- 步骤 6: 径向畸变 ----
        estimateDistortion(views, H, res.K, res.k1, res.k2);

        res.ok = true;
        res.rms = reprojectionRMS(views, res);

        // ---- 步骤 7: LM 精化 ----
        if (doRefine) {
            refineLM(views, res);
            res.rms = reprojectionRMS(views, res);
        }

        res.message = "OK";
    } catch (const std::exception& e) {
        res.ok = false;
        res.message = e.what();
    }

    return res;
}
