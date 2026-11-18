#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_jacobian.py  --  用有限差分检验解析雅可比

LM 精化能不能收敛，完全取决于雅可比对不对。这里不靠"看起来能跑"
来判断，而是直接拿中心差分数值导数逐列比对。

同时做两组端到端检验：
  检验 A  无噪声无畸变   -> 线性闭式解应恢复到机器精度
  检验 B  0.3px 噪声 + 径向畸变 -> LM 应显著降低重投影误差

全部通过退出码 0，否则 1。

运行：  python tools/verify_jacobian.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zhang_reference as zr

FAIL = []


def check(name, ok, detail=""):
    tag = "[OK]  " if ok else "[FAIL]"
    print("%s %s%s" % (tag, name, ("   " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def make_scene():
    K = np.array([[1200.0, 5.0, 640.0],
                  [0.0, 1150.0, 480.0],
                  [0.0, 0.0, 1.0]])
    board = zr.make_board(7, 7, 30.0)
    poses = [
        (zr.rot_y(np.deg2rad(25)), np.array([-60.0, -40.0, 500.0])),
        (zr.rot_x(np.deg2rad(-20)) @ zr.rot_y(np.deg2rad(15)), np.array([80.0, -30.0, 460.0])),
        (zr.rot_z(np.deg2rad(18)) @ zr.rot_x(np.deg2rad(22)), np.array([-40.0, 90.0, 520.0])),
    ]
    return K, board, poses


# ---------------------------------------------------------------------------
#  1. 有限差分 vs 解析雅可比
# ---------------------------------------------------------------------------
def test_jacobian():
    print("=" * 72)
    print(" 检验 1  解析雅可比 vs 中心差分数值导数")
    print("=" * 72)

    K, board, poses = make_scene()
    k1, k2 = -0.08, 0.02

    views = []
    for R, t in poses:
        pr = zr.project_full(K, R, t, board, k1, k2)
        img = np.array([[p['u'], p['v']] for p in pr])
        views.append({'obj': board[:, :2].copy(), 'img': img})

    Rs = [np.array(R, float) for R, _ in poses]
    ts = [np.array(t, float) for _, t in poses]
    NP = 7 + 6 * len(views)

    J = zr.build_jacobian(views, K, Rs, ts, k1, k2)

    eps = 1e-6
    Jfd = np.zeros_like(J)

    for j in range(NP):
        rs = []
        for sign in (+1.0, -1.0):
            Kp = K.copy()
            Rp = [r.copy() for r in Rs]
            tp = [x.copy() for x in ts]
            k1p, k2p = k1, k2

            if j < 7:
                if j == 0:   Kp[0, 0] += sign * eps
                elif j == 1: Kp[1, 1] += sign * eps
                elif j == 2: Kp[0, 1] += sign * eps
                elif j == 3: Kp[0, 2] += sign * eps
                elif j == 4: Kp[1, 2] += sign * eps
                elif j == 5: k1p += sign * eps
                elif j == 6: k2p += sign * eps
            else:
                vi = (j - 7) // 6
                off = (j - 7) % 6
                if off < 3:
                    dw = np.zeros(3)
                    dw[off] = sign * eps
                    Rp[vi] = zr.rodrigues(dw) @ Rp[vi]
                else:
                    dt = np.zeros(3)
                    dt[off - 3] = sign * eps
                    tp[vi] = tp[vi] + dt

            rs.append(zr.residual_vector(views, Kp, Rp, tp, k1p, k2p))

        Jfd[:, j] = (rs[0] - rs[1]) / (2.0 * eps)

    denom = np.maximum(np.abs(Jfd), 1.0)
    rel = np.abs(J - Jfd) / denom
    max_rel = float(rel.max())
    max_abs = float(np.abs(J - Jfd).max())

    print("  参数个数      : %d  (7 内参+畸变 + %d 图 x 6)" % (NP, len(views)))
    print("  残差维度      : %d" % J.shape[0])
    print("  雅可比尺寸    : %s" % (J.shape,))
    print("  最大绝对误差  : %.3e" % max_abs)
    print("  最大相对误差  : %.3e" % max_rel)

    # 逐块报告，便于定位是哪个块写错了
    labels = ['d_alpha', 'd_beta', 'd_gamma', 'd_u0', 'd_v0', 'd_k1', 'd_k2']
    worst = []
    for j in range(7):
        worst.append((labels[j], float(rel[:, j].max())))
    for i in range(len(views)):
        for k, nm in enumerate(['w0', 'w1', 'w2', 't0', 't1', 't2']):
            worst.append(("view%d.%s" % (i + 1, nm), float(rel[:, 7 + 6*i + k].max())))
    worst.sort(key=lambda x: -x[1])
    print("  误差最大的 5 列:")
    for nm, v in worst[:5]:
        print("      %-12s %.3e" % (nm, v))

    check("雅可比与有限差分一致（相对误差 < 1e-6）", max_rel < 1e-6,
          "max_rel = %.3e" % max_rel)
    return max_rel


# ---------------------------------------------------------------------------
#  2. 无噪声、无畸变：应恢复到机器精度
# ---------------------------------------------------------------------------
def test_clean():
    print()
    print("=" * 72)
    print(" 检验 2  无噪声、无畸变 —— 闭式解应达机器精度")
    print("=" * 72)

    K_true = np.array([[1200.0, 0.0, 640.0],
                       [0.0, 1150.0, 480.0],
                       [0.0, 0.0, 1.0]])
    board = zr.make_board(7, 7, 30.0)
    poses = [
        (zr.rot_y(np.deg2rad(25)), np.array([-60.0, -40.0, 500.0])),
        (zr.rot_x(np.deg2rad(-20)) @ zr.rot_y(np.deg2rad(15)), np.array([80.0, -30.0, 460.0])),
        (zr.rot_z(np.deg2rad(18)) @ zr.rot_x(np.deg2rad(22)), np.array([-40.0, 90.0, 520.0])),
        (zr.rot_y(np.deg2rad(-30)) @ zr.rot_x(np.deg2rad(-15)), np.array([120.0, 60.0, 480.0])),
        (zr.rot_z(np.deg2rad(-25)) @ zr.rot_y(np.deg2rad(32)), np.array([10.0, -70.0, 550.0])),
    ]

    views = []
    for R, t in poses:
        img = zr.project(K_true, R, t, board)
        views.append({'obj': board[:, :2].copy(), 'img': img})

    K_est, Rs, ts, _ = zr.calibrate(views)

    e = np.array([abs(K_est[0, 0] - K_true[0, 0]),
                  abs(K_est[1, 1] - K_true[1, 1]),
                  abs(K_est[0, 1] - K_true[0, 1]),
                  abs(K_est[0, 2] - K_true[0, 2]),
                  abs(K_est[1, 2] - K_true[1, 2])])

    rms = 0.0
    n = 0
    for i, v in enumerate(views):
        pr = zr.project_full(K_est, Rs[i], ts[i], v['obj'], 0.0, 0.0)
        for p, (uo, vo) in zip(pr, v['img']):
            rms += (uo - p['u']) ** 2 + (vo - p['v']) ** 2
            n += 2
    rms = float(np.sqrt(rms / n))

    print("  内参绝对误差: alpha %.3e  beta %.3e  gamma %.3e  u0 %.3e  v0 %.3e"
          % tuple(e))
    print("  重投影 RMS  : %.3e 像素" % rms)

    # 旋转矩阵正交性
    worst_orth = 0.0
    worst_det = 0.0
    for R in Rs:
        worst_orth = max(worst_orth, float(np.abs(R.T @ R - np.eye(3)).max()))
        worst_det = max(worst_det, abs(float(np.linalg.det(R)) - 1.0))
    print("  max|R^T R - I| : %.3e" % worst_orth)
    print("  max|det(R) - 1|: %.3e" % worst_det)

    check("内参恢复精度 < 1e-6", float(e.max()) < 1e-6, "max = %.3e" % e.max())
    check("重投影 RMS < 1e-6 像素", rms < 1e-6, "rms = %.3e" % rms)
    check("旋转矩阵正交 < 1e-9", worst_orth < 1e-9, "%.3e" % worst_orth)
    check("det(R) = 1 < 1e-9", worst_det < 1e-9, "%.3e" % worst_det)
    return rms


# ---------------------------------------------------------------------------
#  3. 有噪声 + 有畸变：LM 应显著改善
# ---------------------------------------------------------------------------
def test_noisy_distorted():
    print()
    print("=" * 72)
    print(" 检验 3  0.3 像素噪声 + 径向畸变 —— LM 精化应显著降低误差")
    print("=" * 72)

    K_true = np.array([[1200.0, 0.0, 640.0],
                       [0.0, 1150.0, 480.0],
                       [0.0, 0.0, 1.0]])
    board = zr.make_board(7, 7, 30.0)
    K1_TRUE, K2_TRUE, SIGMA = -0.10, 0.02, 0.3

    # 关键：标定板要有明显不同的距离和倾角，
    # 否则焦距和畸变系数高度相关，问题本身不可辨识
    poses = [
        (zr.rot_y(np.deg2rad(25)) @ zr.rot_x(np.deg2rad(-12)), np.array([-60.0, -40.0, 420.0])),
        (zr.rot_x(np.deg2rad(-20)) @ zr.rot_y(np.deg2rad(15)), np.array([80.0, -30.0, 620.0])),
        (zr.rot_z(np.deg2rad(18)) @ zr.rot_x(np.deg2rad(22)), np.array([-40.0, 90.0, 780.0])),
        (zr.rot_y(np.deg2rad(-30)) @ zr.rot_x(np.deg2rad(-15)), np.array([120.0, 60.0, 500.0])),
        (zr.rot_z(np.deg2rad(-25)) @ zr.rot_y(np.deg2rad(32)), np.array([10.0, -70.0, 900.0])),
        (zr.rot_x(np.deg2rad(30)) @ zr.rot_y(np.deg2rad(-18)), np.array([-90.0, 20.0, 660.0])),
    ]

    rng = np.random.default_rng(7)
    views = []
    for R, t in poses:
        pr = zr.project_full(K_true, R, t, board, K1_TRUE, K2_TRUE)
        img = np.array([[p['u'], p['v']] for p in pr])
        img = img + rng.normal(0.0, SIGMA, img.shape)
        views.append({'obj': board[:, :2].copy(), 'img': img})

    def gRMS(K, Rs, ts, k1=0.0, k2=0.0):
        s, n = 0.0, 0
        for i, v in enumerate(views):
            pr = zr.project_full(K, Rs[i], ts[i], v['obj'], k1, k2)
            for p, (uo, vo) in zip(pr, v['img']):
                s += (uo - p['u']) ** 2 + (vo - p['v']) ** 2
                n += 2
        return float(np.sqrt(s / n))

    K_lin, Rs, ts, _ = zr.calibrate(views)
    rms_lin = gRMS(K_lin, Rs, ts)

    K_lm = np.array(K_lin, float)
    R_lm = [np.array(r, float) for r in Rs]
    t_lm = [np.array(x, float) for x in ts]
    K_lm, R_lm, t_lm, k1, k2, _ = zr.refine_lm(views, K_lm, R_lm, t_lm, iters=200)
    rms_lm = gRMS(K_lm, R_lm, t_lm, k1, k2)

    print("  真值      : alpha 1200  beta 1150  u0 640  v0 480   k1 %+.3f  k2 %+.3f"
          % (K1_TRUE, K2_TRUE))
    print("  线性闭式解: alpha %8.2f  beta %8.2f  u0 %7.2f  v0 %7.2f   RMS %.4f px"
          % (K_lin[0, 0], K_lin[1, 1], K_lin[0, 2], K_lin[1, 2], rms_lin))
    print("  LM 精化后 : alpha %8.2f  beta %8.2f  u0 %7.2f  v0 %7.2f   RMS %.4f px"
          % (K_lm[0, 0], K_lm[1, 1], K_lm[0, 2], K_lm[1, 2], rms_lm))
    print("  畸变估计  : k1 %+.4f (真值 %+.3f)   k2 %+.4f (真值 %+.3f)"
          % (k1, K1_TRUE, k2, K2_TRUE))
    print("  噪声下限  : %.3f px  （注入的高斯噪声水平）" % SIGMA)
    print("  改善倍数  : %.1f 倍" % (rms_lin / max(rms_lm, 1e-12)))

    check("LM 后 RMS 比线性解降低至少 2 倍", rms_lm < rms_lin / 2.0,
          "%.4f -> %.4f" % (rms_lin, rms_lm))
    check("LM 后 RMS 接近噪声下限（< 2x sigma）", rms_lm < 2.0 * SIGMA,
          "%.4f vs 噪声 %.3f" % (rms_lm, SIGMA))
    check("畸变系数符号正确且量级接近",
          abs(k1 - K1_TRUE) < 0.06 and k1 * K1_TRUE > 0,
          "k1 = %+.4f，真值 %+.3f" % (k1, K1_TRUE))
    return rms_lin, rms_lm


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print()
    test_jacobian()
    test_clean()
    test_noisy_distorted()

    print()
    print("=" * 72)
    if FAIL:
        print(" 结果：%d 项未通过" % len(FAIL))
        for f in FAIL:
            print("    - %s" % f)
        print("=" * 72)
        sys.exit(1)
    print(" 结果：全部通过")
    print("=" * 72)
    sys.exit(0)
