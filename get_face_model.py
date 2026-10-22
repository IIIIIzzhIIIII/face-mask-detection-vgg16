#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_face_model.py —— 获取人脸检测模型文件 haarcascade_frontalface_default.xml

背景：OpenCV 5 起安装包不再自带 Haar 级联分类器文件，而本系统的"人脸框标注"功能需要它。
本脚本会自动从国内 PyPI 镜像下载一个自带该文件的 OpenCV 4.x 安装包（whl），
只把里面的 XML 提取到 code\\models\\ 目录，不会改动已安装的 OpenCV 版本。

不运行本脚本也不影响系统使用：inference.py 会自动退化为"对整幅画面做口罩识别"。

用法：
    py -3.12 get_face_model.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TARGET = "haarcascade_frontalface_default.xml"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
# 这个版本的 whl 里带有 cv2/data/*.xml（脚本只取其中的 XML，不会安装它）
SRC_WHL = "opencv-python==4.10.0.84"
EXTRA_XML = ["haarcascade_frontalface_alt.xml", "haarcascade_frontalface_alt2.xml",
             "haarcascade_profileface.xml", "haarcascade_eye.xml"]


def already_have():
    cands = [os.path.join(MODEL_DIR, TARGET)]
    try:
        import cv2
        cands.append(os.path.join(cv2.data.haarcascades, TARGET))
    except Exception:
        pass
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def download_wheel(tmp_dir):
    print("正在从国内镜像下载 %s（约 40MB，仅用于提取 XML）..." % SRC_WHL)
    cmd = [sys.executable, "-m", "pip", "download", SRC_WHL, "--no-deps",
           "--only-binary=:all:", "-d", tmp_dir, "-i", MIRROR, "--progress-bar", "off"]
    subprocess.check_call(cmd)
    whls = glob.glob(os.path.join(tmp_dir, "opencv_python-*.whl"))
    if not whls:
        raise RuntimeError("没有下载到 whl 文件")
    return whls[0]


def extract(whl_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = [TARGET] + EXTRA_XML
    got = []
    with zipfile.ZipFile(whl_path) as z:
        for member in z.namelist():
            base = os.path.basename(member)
            if base in names and (member.endswith(".xml")):
                with z.open(member) as src, open(os.path.join(out_dir, base), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                got.append(base)
    return got


def main():
    exist = already_have()
    if exist:
        print("[跳过] 已经有人脸检测模型文件：%s" % exist)
        return
    tmp_dir = os.path.join(tempfile.gettempdir(), "opencv_whl_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        whl = download_wheel(tmp_dir)
        got = extract(whl, MODEL_DIR)
    except subprocess.CalledProcessError as e:
        print("[失败] 下载出错：%s" % e)
        print("       可手动下载 %s 并解压出 cv2/data/%s 放到 %s" % (SRC_WHL, TARGET, MODEL_DIR))
        return
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if TARGET in got:
        print("[完成] 已提取 %d 个文件到 %s" % (len(got), MODEL_DIR))
        for g in got:
            print("       - %s" % g)
        print("\n现在运行 camera_demo.py / app.py 就能看到人脸检测框了。")
    else:
        print("[失败] whl 里没有找到 %s，请检查 OpenCV 版本" % TARGET)


if __name__ == "__main__":
    main()
