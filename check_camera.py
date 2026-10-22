#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_camera.py —— 摄像头自检（答辩前先跑这个，确认摄像头能不能用）

作用：
  1. 打印 OpenCV 版本、可用后端
  2. 依次尝试 0~4 号摄像头设备，能取到画面就保存截图到 outputs/camera_check_*.jpg
  3. 明确告诉你"摄像头是否可用"，不可用时该怎么做

用法：
    py -3.12 check_camera.py
    py -3.12 check_camera.py --max-index 3
"""
import argparse
import os
import sys

import cv2

from utils import OUT_DIR, ensure_dirs


def list_backends():
    """列出可用的视频后端（不同 OpenCV 版本返回类型不同，这里统一成字符串）"""
    try:
        reg = cv2.videoio_registry
        backs = reg.getBackends()
        if isinstance(backs, (tuple, list)):
            return ", ".join(str(b) for b in backs)
        return str(backs)
    except Exception as e:
        return "(当前 OpenCV 版本不支持查询：%s)" % type(e).__name__


def try_open(index, backend=None, warmup=8):
    """尝试打开摄像头并抓一帧，成功返回 (True, 帧, 分辨率)，失败返回 (False, None, 说明)"""
    cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return False, None, "打不开设备"
    frame = None
    for _ in range(warmup):                     # 摄像头刚打开时前几帧可能是黑的
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        cv2.waitKey(30)
    cap.release()
    if frame is None:
        return False, None, "设备已打开但读取不到画面"
    return True, frame, "%dx%d" % (frame.shape[1], frame.shape[0])


def main():
    ap = argparse.ArgumentParser(description="摄像头自检")
    ap.add_argument("--max-index", type=int, default=4, help="最多检测到几号摄像头设备")
    ap.add_argument("--backend", default="dshow", choices=["dshow", "msmf", "any"],
                    help="Windows 下推荐 dshow")
    args = ap.parse_args()
    ensure_dirs()

    print("=" * 62)
    print("OpenCV 版本：%s" % cv2.__version__)
    print("可用视频后端：%s" % list_backends())
    print("=" * 62)

    backends = {"dshow": getattr(cv2, "CAP_DSHOW", 0),
                "msmf": getattr(cv2, "CAP_MSMF", 0),
                "any": None}
    backend = backends[args.backend]
    found = None
    for idx in range(args.max_index + 1):
        ok, frame, info = try_open(idx, backend)
        print("摄像头索引 %d（后端 %s）：%s  %s" % (idx, args.backend, "可用" if ok else "不可用", info))
        if ok:
            path = os.path.join(OUT_DIR, "camera_check_%d.jpg" % idx)
            cv2.imencode(".jpg", frame)[1].tofile(path)      # 支持中文路径
            print("   已保存截图：%s" % path)
            if found is None:
                found = idx
            break

    print("-" * 62)
    if found is not None:
        print("[OK] 摄像头可用，索引 = %d  -> 运行： py -3.12 camera_demo.py --source %d" % (found, found))
    else:
        print("[提示] 没有检测到可用摄像头。常见原因与处理办法：")
        print("  1) 设备管理器里没有'摄像头'（本机之前确认过属于这种情况：硬件未识别）")
        print("     -> 尝试：Win+X 打开设备管理器 -> 扫描硬件改动；检查笔记本侧面摄像头物理开关")
        print("     -> 也可以在'设置-隐私和安全性-摄像头'里确认权限已开启")
        print("  2) 摄像头被其它程序占用（如腾讯会议、相机应用）-> 关掉其它程序再试")
        print("  3) 驱动被禁用（设备管理器里带黄色感叹号）-> 右键启用 / 重装厂商驱动")
        print("  4) 实在用不了也没关系，本系统支持其它输入源，效果完全一样：")
        print("     py -3.12 camera_demo.py --source demostream    # 用测试图片模拟实时视频（默认）")
        print("     py -3.12 camera_demo.py --source demo_images\\xxx.mp4   # 用手机录的 mp4")
        print("     py -3.12 camera_demo.py --source 手机USB摄像头索引(如 1)")
        print("     py -3.12 camera_demo.py --source screen        # 识别屏幕画面")
        sys.exit(0)


if __name__ == "__main__":
    main()
