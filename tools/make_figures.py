#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py  --  生成 README 用的配图

产出（都写到 docs/images/）:
  calibration_result.svg   2x2 标定结果图：场景、各阶段误差、误差散点、误差直方图
  terminal_output.svg      把 zhang_reference.py 的真实输出渲染成终端窗口

所有数字都来自实际运行，没有手填。

测试场景（两级）:
  场景 A  无噪声、无畸变      -> 检验算法正确性，误差应降到机器精度
  场景 B  0.3px 噪声 + 径向畸变 k1=-0.2, k2=0.05
                              -> 线性闭式解忽略畸变会明显偏差，LM 精化应大幅降低

运行：  python tools/make_figures.py
"""

import os
import io
import sys
import contextlib
import warnings
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zhang_reference as zr

warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'

OUT_DIR = os.path.join(HERE, '..', 'docs', 'images')
os.makedirs(OUT_DIR, exist_ok=True)

C_DARK  = '#1f2937'
C_BLUE  = '#2563eb'
C_RED   = '#dc2626'
C_GREEN = '#059669'
C_AMBER = '#d97706'
C_GRAY  = '#9ca3af'


# ===========================================================================
#  场景与数据
# ===========================================================================
def build_scene():
    K_true = np.array([[1200.0, 0.0, 640.0],
                       [0.0, 1150.0, 480.0],
                       [0.0, 0.0, 1.0]])
    board = zr.make_board(7, 7, 30.0)
    # 距离和倾角都要拉开：否则焦距与畸变系数高度相关，问题本身不可辨识
    poses = [
        (zr.rot_y(np.deg2rad(25)) @ zr.rot_x(np.deg2rad(-12)), np.array([-60.0, -40.0, 420.0])),
        (zr.rot_x(np.deg2rad(-20)) @ zr.rot_y(np.deg2rad(15)), np.array([80.0, -30.0, 620.0])),
        (zr.rot_z(np.deg2rad(18)) @ zr.rot_x(np.deg2rad(22)), np.array([-40.0, 90.0, 780.0])),
        (zr.rot_y(np.deg2rad(-30)) @ zr.rot_x(np.deg2rad(-15)), np.array([120.0, 60.0, 500.0])),
        (zr.rot_z(np.deg2rad(-25)) @ zr.rot_y(np.deg2rad(32)), np.array([10.0, -70.0, 900.0])),
        (zr.rot_x(np.deg2rad(30)) @ zr.rot_y(np.deg2rad(-18)), np.array([-90.0, 20.0, 660.0])),
    ]
    return K_true, board, poses


def make_views(K_true, board, poses, sigma=0.0, k1=0.0, k2=0.0, seed=0):
    """按真值投影出角点；可选加畸变和像素噪声"""
    rng = np.random.default_rng(seed)
    views = []
    for R, t in poses:
        pr = zr.project_full(K_true, R, t, board, k1, k2)
        img = np.array([[p['u'], p['v']] for p in pr])
        if sigma > 0:
            img = img + rng.normal(0.0, sigma, img.shape)
        views.append({'obj': board[:, :2].copy(), 'img': img})
    return views


def per_view_rms(views, K, Rs, ts, k1=0.0, k2=0.0):
    out = []
    for i, v in enumerate(views):
        pr = zr.project_full(K, Rs[i], ts[i], v['obj'], k1, k2)
        s = 0.0
        for p, (uo, vo) in zip(pr, v['img']):
            s += (uo - p['u']) ** 2 + (vo - p['v']) ** 2
        out.append(float(np.sqrt(s / (2 * len(pr)))))
    return np.array(out)


def global_rms(views, K, Rs, ts, k1=0.0, k2=0.0):
    e = per_view_rms(views, K, Rs, ts, k1, k2)
    n = sum(len(v['img']) for v in views)
    return float(np.sqrt((e ** 2 * np.array([len(v['img']) for v in views])).sum() / n))


def run_pipeline(views):
    """线性闭式解 -> LM 精化"""
    K_lin, Rs, ts, _ = zr.calibrate(views)
    rms_lin = global_rms(views, K_lin, Rs, ts)

    K_lm = np.array(K_lin, float)
    R_lm = [np.array(r, float) for r in Rs]
    t_lm = [np.array(x, float) for x in ts]
    K_lm, R_lm, t_lm, k1, k2, _ = zr.refine_lm(views, K_lm, R_lm, t_lm, iters=200)
    rms_lm = global_rms(views, K_lm, R_lm, t_lm, k1, k2)
    return dict(K_lin=K_lin, K_lm=K_lm, R=R_lm, t=t_lm,
                rms_lin=rms_lin, rms_lm=rms_lm, k1=k1, k2=k2)


# ===========================================================================
#  图 1：标定结果
# ===========================================================================
def make_result_figure():
    K_true, board, poses = build_scene()
    K1_TRUE, K2_TRUE = -0.10, 0.02
    SIGMA = 0.3

    views_clean = make_views(K_true, board, poses, sigma=0.0)
    views_hard = make_views(K_true, board, poses, sigma=SIGMA,
                            k1=K1_TRUE, k2=K2_TRUE, seed=7)

    A = run_pipeline(views_clean)     # 无噪声无畸变
    B = run_pipeline(views_hard)      # 有噪声有畸变

    print("\n" + "=" * 72)
    print(" 实际计算结果")
    print("=" * 72)
    print("\n[场景 A]  无噪声、无畸变")
    print("   线性闭式解 RMS = %.3e 像素" % A['rms_lin'])
    print("   LM 精化   RMS = %.3e 像素" % A['rms_lm'])
    print("   估计 K =")
    for row in A['K_lm']:
        print("     [%12.6f %12.6f %12.6f]" % tuple(row))

    print("\n[场景 B]  %.1f 像素噪声 + 畸变 k1=%.2f k2=%.2f" % (SIGMA, K1_TRUE, K2_TRUE))
    print("   线性闭式解 RMS = %.4f 像素   (闭式解假设无畸变)" % B['rms_lin'])
    print("   LM 精化   RMS = %.4f 像素   (降低 %.1f 倍)"
          % (B['rms_lm'], B['rms_lin'] / max(B['rms_lm'], 1e-12)))
    print("   估计 K =")
    for row in B['K_lm']:
        print("     [%12.4f %12.4f %12.4f]" % tuple(row))
    print("   畸变估计 k1 = %+.5f   k2 = %+.5f   (真值 %+.2f / %+.2f)"
          % (B['k1'], B['k2'], K1_TRUE, K2_TRUE))
    e = np.array([B['K_lm'][0, 0] - K_true[0, 0], B['K_lm'][1, 1] - K_true[1, 1],
                  B['K_lm'][0, 2] - K_true[0, 2], B['K_lm'][1, 2] - K_true[1, 2]])
    print("   内参绝对误差 alpha %+.3f  beta %+.3f  u0 %+.3f  v0 %+.3f" % tuple(e))

    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(13.2, 9.4), dpi=110)
    fig.patch.set_facecolor('white')

    # ============ (a) 3D 场景 ============
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    S = 30.0
    quad = [[0, 0, 0], [6*S, 0, 0], [6*S, 6*S, 0], [0, 6*S, 0]]
    ax.add_collection3d(Poly3DCollection([quad], alpha=0.16, facecolor=C_BLUE,
                                         edgecolor=C_DARK, linewidth=1.2))
    for i in range(8):
        ax.plot([i*S, i*S], [0, 6*S], [0, 0], color=C_DARK, lw=0.6, alpha=0.45)
        ax.plot([0, 6*S], [i*S, i*S], [0, 0], color=C_DARK, lw=0.6, alpha=0.45)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(poses)))
    for i, (R, t) in enumerate(poses):
        C = -R.T @ t
        ax.scatter(*C, s=80, color=colors[i], depthshade=False,
                   edgecolor='white', linewidth=1.0, zorder=5)
        # 光轴：指向板中心，长度取相机到板距离的 28%
        d = np.array([3*S, 3*S, 0.0]) - C
        d = d / np.linalg.norm(d) * (np.linalg.norm(d) * 0.28)
        ax.plot([C[0], C[0]+d[0]], [C[1], C[1]+d[1]], [C[2], C[2]+d[2]],
                color=colors[i], lw=1.7, alpha=0.85)
        ax.text(C[0], C[1], C[2] + 55, "  #%d" % (i+1), color=colors[i],
                fontsize=9.5, fontweight='bold')

    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Y (mm)", fontsize=9)
    ax.set_zlabel("Z (mm)", fontsize=9)
    ax.set_title("(a) 合成标定场景：7x7 标定板 + %d 个相机位姿" % len(poses),
                 fontsize=11, fontweight='bold', color=C_DARK, pad=10)
    ax.tick_params(labelsize=8)
    # 固定坐标范围。注意 t 是"标定板在相机坐标系下的位置"，所以相机中心的
    # 世界坐标 C = -R^T t 的 Z 分量是负的（板在 z=0，相机在 z<0）。
    # 若按直觉把 zlim 写成向上为正，相机就全被截到画面外了。
    ax.set_xlim(-430, 530)
    ax.set_ylim(-440, 300)
    ax.set_zlim(-880, 150)
    # 刻度给稀疏一点，否则三个轴的标签会挤成一团
    ax.set_xticks([-400, -200, 0, 200, 400])
    ax.set_yticks([-400, -200, 0, 200])
    ax.set_zticks([-800, -600, -400, -200, 0])
    ax.view_init(elev=17, azim=-56)
    try:
        ax.set_box_aspect((1.0, 0.77, 1.05))
    except Exception:
        pass

    # ============ (b) 各视图 RMS：线性 vs LM ============
    ax = fig.add_subplot(2, 2, 2)
    ev_lin = per_view_rms(views_hard, B['K_lin'],
                          [np.array(r, float) for r in zr.calibrate(views_hard)[1]],
                          [np.array(x, float) for x in zr.calibrate(views_hard)[2]])
    ev_lm = per_view_rms(views_hard, B['K_lm'], B['R'], B['t'], B['k1'], B['k2'])

    x = np.arange(len(poses))
    w = 0.38
    b1 = ax.bar(x - w/2, ev_lin, w, label="线性闭式解（忽略畸变）",
                color=C_AMBER, alpha=0.9)
    b2 = ax.bar(x + w/2, ev_lm, w, label="LM 精化后", color=C_GREEN, alpha=0.9)
    ax.bar_label(b1, fmt='%.2f', fontsize=8.5, padding=2, color=C_AMBER)
    ax.bar_label(b2, fmt='%.2f', fontsize=8.5, padding=2, color=C_GREEN)

    ax.set_xticks(x)
    ax.set_xticklabels(["#%d" % (i+1) for i in range(len(poses))], fontsize=9)
    ax.set_xlabel("视图编号", fontsize=9)
    ax.set_ylabel("重投影误差 RMS（像素）", fontsize=9)
    ax.set_title("(b) 各视图重投影误差：%.0f 倍改善"
                 % (B['rms_lin'] / max(B['rms_lm'], 1e-12)),
                 fontsize=11, fontweight='bold', color=C_DARK, pad=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ev_lin.max() * 1.25)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ============ (c) 误差散点 ============
    ax = fig.add_subplot(2, 2, 3)
    du, dv = [], []
    for i, v in enumerate(views_hard):
        pr = zr.project_full(B['K_lm'], B['R'][i], B['t'][i], v['obj'], B['k1'], B['k2'])
        for p, (uo, vo) in zip(pr, v['img']):
            du.append(uo - p['u']); dv.append(vo - p['v'])
    du, dv = np.array(du), np.array(dv)

    lim = max(1.0, np.abs(np.concatenate([du, dv])).max() * 1.15)
    ax.axhline(0, color=C_GRAY, lw=0.8, zorder=1)
    ax.axvline(0, color=C_GRAY, lw=0.8, zorder=1)
    th = np.linspace(0, 2*np.pi, 200)
    for r, ls in [(0.5, ':'), (1.0, '--')]:
        ax.plot(r*np.cos(th), r*np.sin(th), ls, color=C_GRAY, lw=0.8, zorder=1)
    ax.scatter(du, dv, s=16, alpha=0.55, color=C_BLUE, edgecolor='none', zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel("u 方向误差（像素）", fontsize=9)
    ax.set_ylabel("v 方向误差（像素）", fontsize=9)
    ax.set_title("(c) LM 精化后 %d 个点的重投影误差" % len(du), fontsize=11,
                 fontweight='bold', color=C_DARK, pad=10)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.text(0.02, 0.97, "虚线圆：0.5 / 1.0 像素", transform=ax.transAxes,
            ha='left', va='top', fontsize=8, color=C_GRAY)

    # ============ (d) 误差直方图 ============
    ax = fig.add_subplot(2, 2, 4)
    mag = np.sqrt(du**2 + dv**2)
    ax.hist(mag, bins=26, color=C_BLUE, alpha=0.75, edgecolor='white')
    ax.axvline(mag.mean(), color=C_RED, lw=2.0,
               label="均值 %.3f 像素" % mag.mean())
    ax.axvline(np.percentile(mag, 95), color=C_GREEN, lw=2.0, ls='--',
               label="95 分位 %.3f 像素" % np.percentile(mag, 95))
    ax.axvline(SIGMA, color=C_GRAY, lw=1.6, ls='-.',
               label="注入噪声水平 %.1f 像素" % SIGMA)
    ax.set_xlabel("单点重投影误差（像素）", fontsize=9)
    ax.set_ylabel("点数", fontsize=9)
    ax.set_title("(d) 单点误差直方图", fontsize=11, fontweight='bold',
                 color=C_DARK, pad=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.suptitle("张氏标定法 · C++/Python 实现验证    "
                 "线性解 %.2f px  →  LM 精化 %.2f px" % (B['rms_lin'], B['rms_lm']),
                 fontsize=14.5, fontweight='bold', color=C_DARK, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.962])

    # 同时出 PNG：兼容性最好，README 里优先用它
    path_png = os.path.join(OUT_DIR, 'calibration_result.png')
    fig.savefig(path_png, format='png', dpi=160, facecolor='white',
                bbox_inches='tight')
    path = os.path.join(OUT_DIR, 'calibration_result.svg')
    fig.savefig(path, format='svg', facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print("\n已生成: %s" % os.path.abspath(path_png))
    print("已生成: %s" % os.path.abspath(path))
    return B


# ===========================================================================
#  图 2：终端输出
# ===========================================================================
def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def display_width(s):
    return sum(2 if ord(c) > 0x2E7F else 1 for c in s)


def make_terminal_svg(text, path, title="zhang_reference.py"):
    lines = text.rstrip('\n').split('\n')
    ncol = max(display_width(l) for l in lines)

    FS, LH, PAD, BAR = 13.0, 18.5, 18.0, 38.0
    CW = FS * 0.6017
    W = PAD * 2 + ncol * CW
    H = BAR + PAD + len(lines) * LH + PAD

    o = []
    o.append('<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" '
             'viewBox="0 0 %.0f %.0f" font-family="Consolas, \'Cascadia Mono\', '
             '\'DejaVu Sans Mono\', \'Microsoft YaHei\', monospace">' % (W, H, W, H))
    o.append('<rect x="0" y="0" width="%.0f" height="%.0f" rx="10" fill="#111827"/>' % (W, H))
    o.append('<rect x="0" y="0" width="%.0f" height="%.0f" rx="10" fill="#1f2937"/>' % (W, BAR))
    o.append('<rect x="0" y="%.0f" width="%.0f" height="10" fill="#1f2937"/>' % (BAR - 10, W))
    for i, col in enumerate(['#ff5f57', '#febc2e', '#28c840']):
        o.append('<circle cx="%.0f" cy="%.0f" r="6.5" fill="%s"/>'
                 % (PAD + 8 + i * 20, BAR / 2, col))
    o.append('<text x="%.0f" y="%.0f" fill="#9ca3af" font-size="13.5" '
             'text-anchor="middle">%s</text>' % (W / 2, BAR / 2 + 5, esc(title)))

    y = BAR + PAD + FS
    for ln in lines:
        if not ln.strip():
            y += LH
            continue
        color = '#e5e7eb'
        if ln.startswith('===') or ln.startswith('---'):
            color = '#6b7280'
        elif '[OK]' in ln or 'PASS' in ln:
            color = '#4ade80'
        elif '[FAIL]' in ln:
            color = '#f87171'
        elif ln.strip().startswith('[') and ']' in ln:
            color = '#60a5fa'
        o.append('<text x="%.0f" y="%.1f" fill="%s" font-size="%.1f" '
                 'xml:space="preserve">%s</text>' % (PAD, y, color, FS, esc(ln)))
        y += LH

    o.append('</svg>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(o))
    return path


def make_terminal_capture():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            zr.main()
        except SystemExit:
            pass
    path = os.path.join(OUT_DIR, 'terminal_output.svg')
    make_terminal_svg(buf.getvalue(), path)
    print("已生成: %s" % os.path.abspath(path))
    return buf.getvalue()


# ===========================================================================
if __name__ == '__main__':
    print("生成配图 ...")
    make_result_figure()
    make_terminal_capture()
    print("\n完成。")
