#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_cudnn.py —— 一键修复飞桨 GPU 缺少 cuDNN 的问题（Windows）

现象：
    RuntimeError: The third-party dynamic library (cudnn64_8.dll) that Paddle
    depends on is not configured correctly. (error code is 126)
原因：Windows 版 paddlepaddle-gpu 安装包不含 cuDNN，需要额外提供 cudnn64_8.dll 及其配套 DLL。

本脚本做两件事：
    1. 检查 GPU 是否已经可用（可用就直接结束）
    2. 不可用时：从国内镜像 pip 安装 nvidia-cudnn-cu11 运行库，
       并把它所在的 bin 目录告诉飞桨（utils.py 会自动加入 DLL 搜索路径）

用法：
    py -3.12 fix_cudnn.py            # 检查 + 自动安装 cuDNN
    py -3.12 fix_cudnn.py --check    # 只检查不安装
"""
import glob
import os
import site
import subprocess
import sys

PY = sys.executable
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
# 需要 cuDNN 8.x（DLL 名为 cudnn64_8.dll，与飞桨 CUDA11.8 版匹配）。
# 不同镜像上可用的具体版本号不同，这里按顺序尝试，任意一个成功即可。
CUDNN_CANDIDATES = [
    "nvidia-cudnn-cu11==8.9.5.29",
    "nvidia-cudnn-cu11==8.9.4.25",
    "nvidia-cudnn-cu11==8.9.4.19",
    "nvidia-cudnn-cu11==8.6.0.163",
    "nvidia-cudnn-cu11<9",
]


def find_cudnn_dlls():
    """在常见位置搜索 cudnn64_8.dll"""
    pats = []
    for sp in site.getsitepackages():
        pats.append(os.path.join(sp, "nvidia", "**", "cudnn64_8.dll"))
    for ver in ["v11.8", "v11.7", "v11.6", "v12.0", "v12.1", "v12.2", "v12.3", "v12.4"]:
        pats.append(os.path.join(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                                 ver, "bin", "cudnn64_8.dll"))
    hits = []
    for p in pats:
        hits += glob.glob(p, recursive=True)
    return sorted(set(hits))


def find_nvidia_bin_dirs():
    """返回 site-packages 下 nvidia/*/bin 目录列表（cuDNN、cuBLAS 等的 DLL 都在这里）"""
    dirs = []
    for sp in site.getsitepackages():
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
    return [d for d in dirs if os.path.isdir(d)]


def link_dlls_into_paddle():
    """
    把 cuDNN / cuBLAS 的 DLL 链接到飞桨的 libs 目录。
    原因：Windows 版飞桨只在"自己安装目录下的 libs"和 CUDA 安装目录里找第三方库，
          放在 site-packages\\nvidia\\... 里它找不到（会报 cublasLt64_11.dll / cudnn64_8.dll 找不到）。
    优先用硬链接（不额外占磁盘空间），失败则复制。
    """
    import paddle
    libs = os.path.join(os.path.dirname(os.path.abspath(paddle.__file__)), "libs")
    os.makedirs(libs, exist_ok=True)
    linked, copied = 0, 0
    for d in find_nvidia_bin_dirs():
        for f in glob.glob(os.path.join(d, "*.dll")):
            tgt = os.path.join(libs, os.path.basename(f))
            if os.path.exists(tgt):
                continue
            try:
                os.link(f, tgt)                     # 硬链接：瞬间完成，不占空间
                linked += 1
            except Exception:
                try:
                    shutil.copy2(f, tgt)
                    copied += 1
                except Exception as e:
                    print("      跳过 %s（%s）" % (os.path.basename(f), e))
    return libs, linked, copied


def try_gpu():
    """返回 (是否真的能用 GPU, 说明)。用一次小卷积做真实试探，能排除"缺少 cuDNN"等假阳性"""
    try:
        import paddle
    except Exception as e:
        return False, "导入飞桨失败：%s" % e
    if not paddle.device.is_compiled_with_cuda():
        return False, "当前安装的是 CPU 版飞桨（若要用 GPU，请安装 paddlepaddle-gpu）"
    import utils
    try:
        n = paddle.device.cuda.device_count()
    except Exception as e:
        return False, str(e).split("\n")[0][:110]
    if n <= 0:
        return False, "未检测到 NVIDIA GPU 设备"
    if utils.cuda_usable():
        return True, "GPU 可用，设备数 %d" % n
    return False, "GPU 设备存在但无法计算（通常是缺少 cuDNN 运行库 cudnn64_8.dll）"


def main():
    check_only = "--check" in sys.argv
    print("=" * 68)
    print("[1/3] 检查可用性")
    import utils                                   # 触发 cuDNN 路径设置
    ok, msg = try_gpu()
    print("      " + msg)
    if ok:
        print("\n[完成] GPU 已经可以正常使用，无需修复。")
        print("       验证：py -3.12 selfcheck.py")
        return

    print("\n[2/3] 查找 cuDNN 运行库（cudnn64_8.dll）")
    hits = find_cudnn_dlls()
    if hits:
        print("      已找到：%s" % hits[0])
        ok, msg = try_gpu()
        print("      重新检查：" + msg)
        if ok:
            print("\n[完成] GPU 已可用（utils.py 会自动加载该目录下的 cuDNN）。")
            return
    else:
        print("      未找到 cudnn64_8.dll")

    if check_only:
        print("\n[提示] 已跳过自动安装。需要修复请运行：py -3.12 fix_cudnn.py")
        return

    print("\n[3/3] 从国内镜像安装 cuDNN 运行库（约 700MB，请耐心等待）")
    installed = False
    for pkg in CUDNN_CANDIDATES:
        cmd = [PY, "-m", "pip", "install", pkg, "-i", MIRROR]
        print("\n      尝试安装 %s ..." % pkg)
        try:
            subprocess.check_call(cmd)
            installed = True
            break
        except subprocess.CalledProcessError:
            print("      该版本不可用，继续尝试下一个 ...")
    if not installed:
        print("\n[失败] 所有候选版本都没装上。可手动执行（把 <版本> 换成上面提示里可用的 8.x 版本）：")
        print("       %s -m pip install nvidia-cudnn-cu11==<版本> -i %s" % (PY, MIRROR))
        return

    hits = find_cudnn_dlls()
    print("\n安装后找到 cuDNN：%s" % (hits[0] if hits else "无"))

    libs, linked, copied = link_dlls_into_paddle()
    print("已把 cuDNN/cuBLAS 的 DLL 链接到飞桨库目录：%s（硬链接 %d 个，复制 %d 个）"
          % (libs, linked, copied))

    # 重新在子进程中验证（确保新加入的 DLL 目录生效）
    code = ("import utils; import paddle, paddle.nn as nn;"
            "paddle.set_device('gpu');"
            "print('GPU_DLL_OK count=', paddle.device.cuda.device_count());"
            "print('CONV_OK', list(nn.Conv2D(3, 4, 3, padding=1)(paddle.randn([2, 3, 16, 16])).shape));"
            "print('LINEAR_OK', list(nn.Linear(8, 2)(paddle.randn([2, 8])).shape))")
    print("正在验证 GPU 能否真正计算 ...")
    subprocess.call([PY, "-u", "-c", code], cwd=os.path.dirname(os.path.abspath(__file__)))
    print("\n[完成] 若上面显示 CONV_OK / LINEAR_OK，说明 GPU 已修复。")
    print("       之后所有脚本都会自动使用 GPU（--device auto 即可）。")


if __name__ == "__main__":
    main()
