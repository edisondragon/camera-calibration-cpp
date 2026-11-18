#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_cpp_output.py  --  把 C++ 程序的输出和 Python 参考实现交叉比对

这是整个 CI 里最有价值的一项检查：
    C++ 是独立写的，Python 也是独立写的，两者必须算出同一个内参矩阵。
    任何一边的公式写错（比如旋转雅可比多减了一个 t、
    或者 LM 阻尼约定混用），都会在这里暴露。

用法:
    python tools/check_cpp_output.py <可执行文件路径> [数据文件]
    python tools/check_cpp_output.py --selftest      # 只测解析逻辑，不跑 C++

无噪声数据上两边都应恢复到机器精度，所以用 1e-6 的严格容差。
"""

import os
import re
import sys
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import zhang_reference as zr


# ---------------------------------------------------------------------------
#  解析 C++ 的输出
# ---------------------------------------------------------------------------
def parse_cpp_output(text):
    """从 C++ 的 stdout 里抠出 K、k1、k2、RMS"""
    out = {}

    m = re.search(r'重投影误差\s*RMS\s*=\s*([-+0-9.eE]+)', text)
    if not m:
        raise ValueError("输出里找不到「重投影误差 RMS」这一行")
    out['rms'] = float(m.group(1))

    m = re.search(r'k1\s*=\s*([-+0-9.eE]+)', text)
    out['k1'] = float(m.group(1)) if m else 0.0
    m = re.search(r'k2\s*=\s*([-+0-9.eE]+)', text)
    out['k2'] = float(m.group(1)) if m else 0.0

    # 内参矩阵：取「【内参矩阵 K】」之后第一段里的三个方括号行
    if '【内参矩阵 K】' not in text:
        raise ValueError("输出里找不到「【内参矩阵 K】」")
    sec = text.split('【内参矩阵 K】', 1)[1]
    sec = re.split(r'【', sec)[0]
    rows = re.findall(
        r'\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]', sec)
    if len(rows) < 3:
        raise ValueError("内参矩阵只解析到 %d 行，期望 3 行" % len(rows))
    out['K'] = np.array([[float(a), float(b), float(c)]
                         for a, b, c in rows[:3]])

    # 各视图误差（可选）
    out['per_view'] = [float(x) for x in
                       re.findall(r'第\s*\d+\s*张:\s*RMS\s*=\s*([-+0-9.eE]+)', text)]
    return out


# ---------------------------------------------------------------------------
#  读取与 C++ 相同的数据文件
# ---------------------------------------------------------------------------
def load_data(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.split('#')[0].strip()
            if ln:
                lines.append(ln)
    nv, npt = (int(x) for x in lines[0].split()[:2])
    idx = 1
    views = []
    for _ in range(nv):
        obj, img = [], []
        for _ in range(npt):
            X, Y, u, v = (float(x) for x in lines[idx].split()[:4])
            idx += 1
            obj.append((X, Y))
            img.append((u, v))
        views.append({'obj': np.array(obj, float),
                      'img': np.array(img, float)})
    return views


# ---------------------------------------------------------------------------
def compare(views, cpp):
    K_lin, Rs, ts, _ = zr.calibrate(views)
    K_py = np.array(K_lin, float)
    R_py = [np.array(r, float) for r in Rs]
    t_py = [np.array(x, float) for x in ts]
    K_py, R_py, t_py, k1, k2, rms_py = zr.refine_lm(
        views, K_py, R_py, t_py, iters=200)

    print("  C++     RMS = %.10e" % cpp['rms'])
    print("  Python  RMS = %.10e" % rms_py)
    print()
    print("  C++     K =")
    for r in cpp['K']:
        print("      [%14.6f %14.6f %14.6f]" % tuple(r))
    print("  Python  K =")
    for r in K_py:
        print("      [%14.6f %14.6f %14.6f]" % tuple(r))

    dK = np.abs(cpp['K'] - K_py)
    print()
    print("  内参最大绝对差 = %.3e" % dK.max())
    print("  C++     k1 = %+.8f   k2 = %+.8f" % (cpp['k1'], cpp['k2']))
    print("  Python  k1 = %+.8f   k2 = %+.8f" % (k1, k2))

    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print("  %s %s%s" % ("[OK]  " if cond else "[FAIL]", name,
                             ("   " + detail) if detail else ""))
        if not cond:
            ok = False

    print()
    chk("C++ 重投影误差 < 1e-6 像素", cpp['rms'] < 1e-6,
        "%.3e" % cpp['rms'])
    # C++ 用 %.4f 打印 K，本身就有 5e-5 的舍入，容差不能比这更紧。
    # 真要是公式写错，差值是几十上百的量级，1e-3 一样抓得住。
    chk("C++ 与 Python 的 K 一致（< 1e-3）", dK.max() < 1e-3,
        "%.3e" % dK.max())
    chk("C++ 与 Python 的 RMS 一致（< 1e-6）",
        abs(cpp['rms'] - rms_py) < 1e-6,
        "%.3e" % abs(cpp['rms'] - rms_py))
    return ok


# ---------------------------------------------------------------------------
def selftest():
    """用一段仿造的 C++ 输出验证解析逻辑"""
    fake = """
        【内参矩阵 K】
   K = [    1200.0000        0.0000      640.0000 ]
          [       0.0000     1150.0000      480.0000 ]
          [       0.0000        0.0000        1.0000 ]

【畸变系数】
   k1 = 0.0000000000
   k2 = 0.0000000000

   重投影误差 RMS = 0.0000000002 像素
     第  1 张:  RMS = 0.000000 像素
     第  2 张:  RMS = 0.000000 像素
"""
    r = parse_cpp_output(fake)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print("  %s %s%s" % ("[OK]  " if cond else "[FAIL]", name,
                             ("   " + detail) if detail else ""))
        if not cond:
            ok = False

    print("  RMS      = %r" % r['rms'])
    print("  K        =\n%s" % r['K'])
    print("  k1, k2   = %r, %r" % (r['k1'], r['k2']))
    print("  各视图   = %r" % r['per_view'])
    print()

    chk("RMS 解析正确", abs(r['rms'] - 2e-10) < 1e-15)
    chk("K 解析正确",
        np.allclose(r['K'], [[1200, 0, 640], [0, 1150, 480], [0, 0, 1]]))
    chk("k1/k2 解析正确", r['k1'] == 0.0 and r['k2'] == 0.0)
    chk("各视图误差解析正确", len(r['per_view']) == 2)
    return ok


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 72)
    if '--selftest' in sys.argv:
        print(" 解析逻辑自测")
        print("=" * 72)
        sys.exit(0 if selftest() else 1)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    exe = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, 'data', 'synthetic.txt')

    print(" C++ 与 Python 参考实现交叉比对")
    print("=" * 72)
    print("  可执行文件: %s" % exe)
    print("  数据文件  : %s" % data)
    print()

    if not os.path.exists(exe):
        print("  [FAIL] 找不到可执行文件")
        sys.exit(1)

    proc = subprocess.run([exe, data], capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        print("  [FAIL] C++ 程序退出码 %d" % proc.returncode)
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        sys.exit(1)

    print(proc.stdout)

    try:
        cpp = parse_cpp_output(proc.stdout)
    except ValueError as e:
        print("  [FAIL] 解析 C++ 输出失败: %s" % e)
        sys.exit(1)

    views = load_data(data)
    ok = compare(views, cpp)

    print("=" * 72)
    print(" 结果：%s" % ("全部通过" if ok else "未通过"))
    print("=" * 72)
    sys.exit(0 if ok else 1)
