#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zhang_reference.py  --  张氏标定法的 Python 参考实现

作用有两个：
  1. 验证 C++ 版本 (src/zhang.cpp) 的算法逻辑是否正确
  2. 生成 C++ 程序可以直接读入的合成测试数据

这份 Python 代码是 src/zhang.cpp 的逐行镜像，
包括 Hartley 归一化、DLT、Vb=0、B 反解内参、SVD 正交化等全部步骤。

运行：  python zhang_reference.py
"""

import numpy as np
import sys
import os

np.set_printoptions(precision=6, suppress=True, linewidth=140)


# ===========================================================================
#  自己实现 Jacobi 特征分解（用于验证 C++ 里那份实现）
# ===========================================================================
def jacobi_eigen(A_in):
    """对称矩阵的 Jacobi 特征分解。返回 (特征值升序, 特征向量按列)"""
    A = np.array(A_in, dtype=float)
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(100):
        off = sum(A[p, q] ** 2 for p in range(n) for q in range(p + 1, n))
        if off < 1e-30:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) < 1e-300:
                    continue
                theta = (A[q, q] - A[p, p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + np.sqrt(theta * theta + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c

                # A <- A * J
                for k in range(n):
                    akp, akq = A[k, p], A[k, q]
                    A[k, p] = c * akp - s * akq
                    A[k, q] = s * akp + c * akq
                # A <- J^T * A
                for k in range(n):
                    apk, aqk = A[p, k], A[q, k]
                    A[p, k] = c * apk - s * aqk
                    A[q, k] = s * apk + c * aqk
                # V <- V * J
                for k in range(n):
                    vkp, vkq = V[k, p], V[k, q]
                    V[k, p] = c * vkp - s * vkq
                    V[k, q] = s * vkp + c * vkq

    ev = np.diag(A).copy()
    order = np.argsort(ev)
    return ev[order], V[:, order]


def smallest_eigen_vector(A):
    """A 的最小奇异向量 = A^T A 的最小特征向量"""
    AtA = A.T @ A
    ev, V = jacobi_eigen(AtA)
    x = V[:, 0]
    return x / np.linalg.norm(x)


# ===========================================================================
#  张氏标定法核心
# ===========================================================================
def normalize_params(pts):
    """点集的归一化参数：重心 + 缩放系数（使平均距离 = sqrt(2)）"""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    d = np.mean(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]))
    s = np.sqrt(2.0) / d
    return c[0], c[1], s


def compute_homography(obj_xy, img_uv):
    """DLT + Hartley 归一化，求单应矩阵"""
    n = len(obj_xy)
    assert n >= 4

    cu, cv, su = normalize_params(img_uv)
    cX, cY, sX = normalize_params(obj_xy)

    A = np.zeros((2 * n, 9))
    for i in range(n):
        X = sX * (obj_xy[i][0] - cX)
        Y = sX * (obj_xy[i][1] - cY)
        u = su * (img_uv[i][0] - cu)
        v = su * (img_uv[i][1] - cv)
        A[2 * i]     = [0, 0, 0, -X, -Y, -1,  v * X,  v * Y,  v]
        A[2 * i + 1] = [X, Y, 1,  0,  0,  0, -u * X, -u * Y, -u]

    h = smallest_eigen_vector(A)
    Hn = h.reshape(3, 3)

    Tinv = np.array([[1 / su, 0, cu],
                     [0, 1 / su, cv],
                     [0, 0, 1.0]])
    S = np.array([[sX, 0, -sX * cX],
                  [0, sX, -sX * cY],
                  [0, 0, 1.0]])

    H = Tinv @ Hn @ S
    if abs(H[2, 2]) > 1e-300:
        H = H / H[2, 2]
    return H


def vij(hi, hj):
    """文档 6.2 节的 v(ij)"""
    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[0] * hj[2] + hi[2] * hj[0],
        hi[1] * hj[2] + hi[2] * hj[1],
        hi[2] * hj[2],
    ])


def K_from_b(b):
    """从 b 反解内参（文档第 7 章）"""
    B11, B12, B22, B13, B23, B33 = b
    denom = B11 * B22 - B12 * B12
    v0 = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / denom)
    gamma = -B12 * alpha * alpha * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha * alpha / lam
    K = np.array([[alpha, gamma, u0],
                  [0, beta, v0],
                  [0, 0, 1.0]])
    return K


def nearest_rotation(Q):
    """把 Q 投影到最近的合法旋转矩阵"""
    QtQ = Q.T @ Q
    lam, V = jacobi_eigen(QtQ)
    flip = np.array([1.0, 1.0, 1.0])
    if np.linalg.det(Q) < 0:
        flip[2] = -1.0
    W = V @ np.diag(flip / np.sqrt(lam)) @ V.T
    return Q @ W


def extrinsics_from_H(H, K):
    """从单应矩阵和外参求 R, t（文档第 8 章）"""
    alpha, gamma, u0 = K[0, 0], K[0, 1], K[0, 2]
    beta, v0 = K[1, 1], K[1, 2]

    Kinv = np.array([[1 / alpha, -gamma / (alpha * beta), (gamma * v0 - beta * u0) / (alpha * beta)],
                     [0, 1 / beta, -v0 / beta],
                     [0, 0, 1.0]])

    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    a1, a2, a3 = Kinv @ h1, Kinv @ h2, Kinv @ h3

    lam = 2.0 / (np.linalg.norm(a1) + np.linalg.norm(a2))
    r1 = a1 * lam
    r2 = a2 * lam
    t = a3 * lam

    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)

    Q = np.column_stack([r1, r2, r3])
    R = nearest_rotation(Q)

    # 消除 H 整体符号的歧义：让标定板位于相机前方
    if t[2] < 0:
        R[:, 0] = -R[:, 0]
        R[:, 1] = -R[:, 1]
        t = -t
    return R, t


def calibrate(views):
    """完整流程"""
    nv = len(views)
    Hs = [compute_homography(v['obj'], v['img']) for v in views]

    V = np.zeros((2 * nv, 6))
    for i, H in enumerate(Hs):
        h1, h2 = H[:, 0], H[:, 1]
        V[2 * i]     = vij(h1, h2)
        V[2 * i + 1] = vij(h1, h1) - vij(h2, h2)

    b = smallest_eigen_vector(V)
    if b[0] < 0:
        b = -b

    K = K_from_b(b)
    Rs, ts = [], []
    for H in Hs:
        R, t = extrinsics_from_H(H, K)
        Rs.append(R)
        ts.append(t)
    return K, Rs, ts, Hs


# ===========================================================================
#  数据生成
# ===========================================================================
def make_board(cols=7, rows=7, square=30.0):
    """生成棋盘格角点的世界坐标（单位 mm，Z=0 平面）"""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append([c * square, r * square, 0.0])
    return np.array(pts)


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])


def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


def project(K, R, t, pts3d, k1=0.0, k2=0.0):
    """把三维点投影到像素（含畸变）"""
    out = []
    for M in pts3d:
        Xc = R @ M + t
        xn, yn = Xc[0] / Xc[2], Xc[1] / Xc[2]
        r2 = xn * xn + yn * yn
        f = 1 + k1 * r2 + k2 * r2 * r2
        xd, yd = xn * f, yn * f
        u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
        v = K[1, 1] * yd + K[1, 2]
        out.append([u, v])
    return np.array(out)


# ===========================================================================
#  主流程
# ===========================================================================
# ===========================================================================
#  非线性精化（Levenberg-Marquardt）—— 与 C++ 的 refineLM 对应
# ===========================================================================
def rodrigues(w):
    """旋转向量 -> 旋转矩阵"""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / th
    Kx = np.array([[0.0, -k[2], k[1]],
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * Kx + (1.0 - np.cos(th)) * (Kx @ Kx)


def project_full(K, R, t, pts3d, k1, k2):
    """完整投影（含畸变）。返回每个点的全部中间量，雅可比要用"""
    alpha, gamma, u0 = K[0, 0], K[0, 1], K[0, 2]
    beta, v0 = K[1, 1], K[1, 2]
    out = []
    for M in pts3d:
        M = np.asarray(M, float)
        if M.size == 2:                       # 板坐标只给了 (X, Y)，补 Z = 0
            M = np.array([M[0], M[1], 0.0])
        Xc = R @ M + t
        xn, yn = Xc[0] / Xc[2], Xc[1] / Xc[2]
        r2 = xn * xn + yn * yn
        f = 1.0 + k1 * r2 + k2 * r2 * r2
        xd, yd = xn * f, yn * f
        out.append(dict(u=alpha * xd + gamma * yd + u0,
                        v=beta * yd + v0,
                        Xc=Xc, xn=xn, yn=yn, r2=r2, f=f, xd=xd, yd=yd))
    return out


def build_jacobian(views, K, R, t, k1, k2):
    """
    解析雅可比矩阵。每行对应一个残差分量，每点两行（u 残差、v 残差）。

    列布局:  [0..6] d_alpha,d_beta,d_gamma,d_u0,d_v0,d_k1,d_k2
             之后每张图 6 列: 旋转增量 dw(3) + 平移增量 dt(3)

    残差定义 r = 观测 - 投影，所以各偏导前面统一带负号。

    链式法则（这是整个 LM 的核心，也是最好出错的地方）:

        du/dw = (du/dxd) · (dxd/dxn) · (dxn/dXc) · (dXc/dw)
                 └Block 1┘   └─Block 2─┘   └Block 3┘   └Block 4┘

    其中 Block 1 是内参矩阵 K 的前两行，已经并入 M3。
    """
    nv = len(views)
    alpha, gamma, u0 = K[0, 0], K[0, 1], K[0, 2]
    beta, v0 = K[1, 1], K[1, 2]
    NP = 7 + 6 * nv
    rows = []

    for i in range(nv):
        pr = project_full(K, R[i], t[i], views[i]['obj'], k1, k2)
        base = 7 + 6 * i
        for p in pr:
            Xc = p['Xc']
            Zc = Xc[2]
            invZ = 1.0 / Zc
            xn, yn, r2, f = p['xn'], p['yn'], p['r2'], p['f']

            # Block 2: (xd, yd) 对 (xn, yn)
            #   xd = xn(1 + k1 r2 + k2 r2^2),  r2 = xn^2 + yn^2
            d1 = f + 2 * k1 * xn * xn + 4 * k2 * xn * xn * r2
            d2 = 2 * k1 * xn * yn + 4 * k2 * xn * yn * r2
            d3 = f + 2 * k1 * yn * yn + 4 * k2 * yn * yn * r2
            B2 = np.array([[d1, d2], [d2, d3]])

            # Block 3: (xn, yn) 对 (Xc, Yc, Zc)
            B3 = np.array([[invZ, 0.0, -Xc[0] * invZ ** 2],
                           [0.0, invZ, -Xc[1] * invZ ** 2]])

            # Block 1 + 2 + 3 合成 2x3（Block 1 就是 K 的前两行）
            M3 = np.array([[alpha, gamma], [0.0, beta]]) @ B2 @ B3

            # Block 4: dXc/dw
            #   Xc = R*X + t, 旋转扰动是左乘: R -> rodrigues(dw) @ R
            #   Xc' = (I + [dw]_x)(R*X) + t = Xc + dw x (R*X)
            #   所以 dXc/dw = [dw]_x 作用在 (R*X) 上, 即 Xc - t。
            #   注意 t 不随旋转变化 —— 写成 Xc 是常见错误, 会让每一列都偏。
            v = Xc - t[i]                      # 就是 R*X
            dXdw = np.array([[0.0, v[2], -v[1]],
                             [-v[2], 0.0, v[0]],
                             [v[1], -v[0], 0.0]])

            row_u = np.zeros(NP)
            row_v = np.zeros(NP)

            # 旋转 + 平移
            row_u[base:base + 3] = -(M3[0] @ dXdw)
            row_v[base:base + 3] = -(M3[1] @ dXdw)
            row_u[base + 3:base + 6] = -M3[0]
            row_v[base + 3:base + 6] = -M3[1]

            # 内参: u = alpha*xd + gamma*yd + u0,  v = beta*yd + v0
            row_u[0:5] = -np.array([p['xd'], 0.0, p['yd'], 1.0, 0.0])
            row_v[0:5] = -np.array([0.0, p['yd'], 0.0, 0.0, 1.0])

            # 畸变系数
            row_u[5] = -(alpha * xn * r2 + gamma * yn * r2)
            row_u[6] = -(alpha * xn * r2 * r2 + gamma * yn * r2 * r2)
            row_v[5] = -(beta * yn * r2)
            row_v[6] = -(beta * yn * r2 * r2)

            rows.append(row_u)
            rows.append(row_v)

    return np.array(rows)


def residual_vector(views, K, R, t, k1, k2):
    """残差向量 r = 观测 - 投影，每点两个分量 (u, v)"""
    out = []
    for i in range(len(views)):
        pr = project_full(K, R[i], t[i], views[i]['obj'], k1, k2)
        for p, (uo, vo) in zip(pr, views[i]['img']):
            out.append(uo - p['u'])
            out.append(vo - p['v'])
    return np.array(out)


def refine_lm(views, K, Rs, ts, k1=0.0, k2=0.0, iters=60):
    """
    Levenberg-Marquardt 精化重投影误差。

    参数向量布局与 C++ 版一致:
        [0..6]  : d_alpha, d_beta, d_gamma, d_u0, d_v0, d_k1, d_k2
        之后每张图 6 个: 旋转增量 dw(3) + 平移增量 dt(3)
    """
    nv = len(views)
    alpha, gamma, u0 = K[0, 0], K[0, 1], K[0, 2]
    beta, v0 = K[1, 1], K[1, 2]
    R = [np.array(r, float) for r in Rs]
    t = [np.array(x, float) for x in ts]
    NP = 7 + 6 * nv

    def residuals():
        return residual_vector(views, K, R, t, k1, k2)

    def jacobian():
        return build_jacobian(views, K, R, t, k1, k2)

    r = residuals()
    F = 0.5 * float(r @ r)

    # 采用 Marquardt 约定: (A + mu * diag(A)) delta = -g
    # mu 是无量纲的，初值取 1e-3。
    # 注意不要写成 mu = 1e-3 * max(diag(A)) —— 那是 Levenberg 的 (A + mu*I)
    # 约定; 两者混用会让阻尼项被 diag(A) 乘两次，步长恒为零。
    mu, nu = 1e-3, 2.0

    for it in range(iters):
        if mu > 1e12:                 # 阻尼大到不可能再改进，收敛退出
            break
        J = jacobian()
        g = J.T @ r
        if np.max(np.abs(g)) < 1e-12:
            break
        A = J.T @ J
        D = np.maximum(np.diag(A).copy(), 1e-12)

        for _ in range(30):
            if mu > 1e12:
                break
            Ad = A + mu * np.diag(D)
            try:
                delta = -np.linalg.solve(Ad, g)
            except np.linalg.LinAlgError:
                mu *= nu; nu *= 2.0; continue

            dL = 0.5 * float(delta @ (mu * D * delta - g))
            if dL <= 0.0:
                mu *= nu; nu *= 2.0; continue

            # 试走一步
            old = (alpha, beta, gamma, u0, v0, k1, k2,
                   [x.copy() for x in R], [x.copy() for x in t])
            alpha += delta[0]; beta += delta[1]; gamma += delta[2]
            u0 += delta[3];    v0 += delta[4]
            k1 += delta[5];    k2 += delta[6]
            K[0, 0], K[0, 1], K[0, 2] = alpha, gamma, u0
            K[1, 1], K[1, 2] = beta, v0
            for i in range(nv):
                base = 7 + 6 * i
                R[i] = rodrigues(delta[base:base + 3]) @ R[i]
                t[i] = t[i] + delta[base + 3:base + 6]

            r_new = residuals()
            F_new = 0.5 * float(r_new @ r_new)
            rho = (F - F_new) / dL

            if rho > 0.0:
                r, F = r_new, F_new
                mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                break
            else:
                (alpha, beta, gamma, u0, v0, k1, k2, R, t) = old
                K[0, 0], K[0, 1], K[0, 2] = alpha, gamma, u0
                K[1, 1], K[1, 2] = beta, v0
                mu *= nu; nu *= 2.0

    rms = float(np.sqrt(np.mean(r ** 2))) if len(r) else 0.0
    return K, R, t, k1, k2, rms


def main():
    print("=" * 74)
    print(" 张氏标定法 Python 参考实现 —— 算法正确性验证")
    print("=" * 74)

    # ---------------- 0. 先单独验证 Jacobi 特征分解 ----------------
    print("\n[验证 0] Jacobi 特征分解 vs numpy.linalg.eigh")
    rng = np.random.default_rng(42)
    S = rng.normal(size=(5, 5))
    Sym = S + S.T                      # 构造对称矩阵
    ev_mine, V_mine = jacobi_eigen(Sym)
    ev_np = np.linalg.eigvalsh(Sym)
    print("   自己实现的特征值 :", ev_mine)
    print("   numpy  的特征值 :", ev_np)
    print("   最大绝对误差     : %.3e" % np.max(np.abs(ev_mine - ev_np)))
    # 检查特征向量满足 A v = lambda v
    res = Sym @ V_mine - V_mine * ev_mine
    print("   特征方程残差     : %.3e" % np.max(np.abs(res)))
    print("   --> %s" % ("通过" if np.max(np.abs(ev_mine - ev_np)) < 1e-8 else "失败"))

    # ---------------- 1. 设定真值 ----------------
    print("\n[验证 1] 设定相机真值")
    K_true = np.array([[1200.0, 0.0, 640.0],
                       [0.0, 1150.0, 480.0],
                       [0.0, 0.0, 1.0]])
    print("   真值 K =")
    print("   " + str(K_true).replace("\n", "\n   "))

    board = make_board(7, 7, 30.0)
    print("   标定板: 7x7 角点, 方格 30mm, 共 %d 个点" % len(board))

    # ---------------- 2. 生成 5 个不同姿态 ----------------
    poses = [
        (rot_y(np.deg2rad(25)), np.array([-60.0, -40.0, 500.0])),
        (rot_x(np.deg2rad(-20)) @ rot_y(np.deg2rad(15)), np.array([80.0, -30.0, 460.0])),
        (rot_z(np.deg2rad(18)) @ rot_x(np.deg2rad(22)), np.array([-40.0, 90.0, 520.0])),
        (rot_y(np.deg2rad(-30)) @ rot_x(np.deg2rad(-15)), np.array([120.0, 60.0, 480.0])),
        (rot_z(np.deg2rad(-25)) @ rot_y(np.deg2rad(32)), np.array([10.0, -70.0, 550.0])),
    ]

    views = []
    for R, t in poses:
        img = project(K_true, R, t, board)
        views.append({'obj': board[:, :2].copy(), 'img': img})

    print("   生成了 %d 张视图（无噪声，用于验证算法正确性）" % len(views))

    # ---------------- 3. 线性闭式解 ----------------
    print("\n[验证 2] 线性闭式解")
    K_est, Rs, ts, Hs = calibrate(views)

    print("   第 1 张图的单应矩阵 H =")
    print("   " + str(Hs[0]).replace("\n", "\n   "))

    print("\n   估计 K =")
    print("   " + str(K_est).replace("\n", "\n   "))

    # 注意：只比较有意义的 5 个参数，不能做整体逐元素相除
    # （K 的 (0,1) 位置真值是 0，除法会出现 0/0）
    names = ['alpha', 'beta ', 'gamma', 'u0   ', 'v0   ']
    got = [K_est[0, 0], K_est[1, 1], K_est[0, 1], K_est[0, 2], K_est[1, 2]]
    tru = [K_true[0, 0], K_true[1, 1], K_true[0, 1], K_true[0, 2], K_true[1, 2]]
    print("\n   绝对误差:")
    for nm, g, tt in zip(names, got, tru):
        print("     %s: 真值 %12.6f   估计 %12.6f   误差 %.3e" % (nm, tt, g, abs(g - tt)))
    err_K = max(abs(g - tt) for g, tt in zip(got, tru))

    # ---------------- 4. 检查外参正交性 ----------------
    print("\n[验证 3] 外参正交性（应该接近单位矩阵）")
    for i, R in enumerate(Rs):
        orth = R.T @ R
        err = np.max(np.abs(orth - np.eye(3)))
        det = np.linalg.det(R)
        print("   第 %d 张: max|R^T R - I| = %.3e ,  det(R) = %.6f" % (i + 1, err, det))

    # ---------------- 5. 重投影误差 ----------------
    print("\n[验证 4] 重投影误差")
    total = 0.0
    cnt = 0
    for i, v in enumerate(views):
        pred = project(K_est, Rs[i], ts[i], board)
        d = pred - v['img']
        total += np.sum(d * d)
        cnt += 2 * len(d)
    rms = np.sqrt(total / cnt)
    print("   重投影误差 RMS = %.3e 像素" % rms)

    # ---------------- 6. 结论 ----------------
    print("\n" + "=" * 74)
    ok = (err_K < 1e-6 and rms < 1e-6)
    if ok:
        print(" 结论: 算法正确 —— 内参完全复原，重投影误差为机器精度")
    else:
        print(" 结论: 存在偏差，需要检查")
    print("=" * 74)

    # ---------------- 7. 导出 C++ 可读的数据文件 ----------------
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'synthetic.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# 合成标定数据（由 tools/zhang_reference.py 生成）\n")
        f.write("# 格式: 第1行 = 视图数 每视图点数\n")
        f.write("#       之后每行 = X Y u v\n")
        f.write("# 真值 K: alpha=1200 beta=1150 u0=640 v0=480\n")
        f.write("%d %d\n" % (len(views), len(board)))
        for v in views:
            for (X, Y, _), (u, vv) in zip(board, v['img']):
                f.write("%.10f %.10f %.10f %.10f\n" % (X, Y, u, vv))
    print("\n 已导出测试数据: %s" % os.path.abspath(path))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
