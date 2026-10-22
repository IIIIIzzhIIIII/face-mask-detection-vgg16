#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_dataset.py —— 自动下载口罩数据集（不需要 Kaggle 账号）

数据来源：HuggingFace 国内镜像 hf-mirror.com 上的公开数据集
         （内容等价于 Kaggle 的 face-mask-12k-images-dataset，约 330MB / 1 万张真实人脸图）

用法：
    py -3.12 download_dataset.py
    py -3.12 download_dataset.py --url <其它直链> --out dataset

特性：
    * 支持断点续传（下载中断后再跑一次会接着下）
    * 自动解压到 dataset/raw/
    * 只依赖标准库，无需先装任何三方包
"""
import argparse
import os
import sys
import time
import zipfile
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 主数据源（国内镜像，实测 HTTP 200 可直连）
DEFAULT_URL = ("https://hf-mirror.com/datasets/ismailfaridi/Face-Mask-Detection-System"
               "/resolve/main/Dataset.zip")
# 备用数据源（161MB，7z 格式，主源不可用时手动指定 --url 后再用 7-Zip 解压）
BACKUP_URL = ("https://hf-mirror.com/datasets/hydramst/face_mask_wearing/resolve/main/dataset.7z")


def human(size):
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return "%.1f %s" % (size, unit)
        size /= 1024.0
    return "%.1f TB" % size


def download(url, out_path, retry=5):
    """带断点续传的下载"""
    for attempt in range(1, retry + 1):
        try:
            exist = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            if exist > 0:
                headers["Range"] = "bytes=%d-" % exist
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=60)
            total = resp.headers.get("Content-Length")
            total = int(total) + exist if total else 0
            mode = "ab" if exist > 0 and resp.status == 206 else "wb"
            if mode == "wb":
                exist = 0
            print("开始下载：%s" % url)
            print("已存在：%s" % human(exist))
            done = exist
            last_report = time.time()
            with open(out_path, mode) as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last_report > 2:
                        last_report = time.time()
                        if total:
                            pct = done * 100.0 / total
                            sys.stdout.write("\r  进度 %.1f%%  %s / %s   " %
                                             (pct, human(done), human(total)))
                        else:
                            sys.stdout.write("\r  已下载 %s   " % human(done))
                        sys.stdout.flush()
            print("")
            if total and done < total:
                raise IOError("下载不完整：%s / %s" % (human(done), human(total)))
            print("[OK] 下载完成：%s（%s）" % (out_path, human(os.path.getsize(out_path))))
            return True
        except Exception as e:
            print("\n[第 %d/%d 次尝试失败] %s: %s" % (attempt, retry, type(e).__name__, e))
            if attempt < retry:
                print("3 秒后重试（支持断点续传）...")
                time.sleep(3)
    return False


def unzip(zip_path, out_dir):
    print("正在解压到：%s ..." % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print("[OK] 解压完成")


def show_tree(root, max_show=40):
    """打印解压后的目录结构（只看目录和图片数量），方便确认数据对不对"""
    print("\n解压后的目录结构（前 %d 项）：" % max_show)
    shown = 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 3:
            continue
        n_img = len([f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        rel = os.path.relpath(dirpath, root)
        print("%s%s/   (图片 %d 张)" % ("  " * depth, rel, n_img))
        shown += 1
        if shown >= max_show:
            print("  ...")
            break


def main():
    ap = argparse.ArgumentParser(description="下载口罩数据集（免 Kaggle 登录）")
    ap.add_argument("--url", default=DEFAULT_URL, help="数据集直链")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "dataset"), help="下载/解压目录")
    ap.add_argument("--backup", action="store_true", help="改用备用数据源（7z，需自行用 7-Zip 解压）")
    args = ap.parse_args()

    url = BACKUP_URL if args.backup else args.url
    os.makedirs(args.out, exist_ok=True)
    ext = ".7z" if url.lower().endswith(".7z") else ".zip"
    zip_path = os.path.join(args.out, "Dataset" + ext)
    raw_dir = os.path.join(args.out, "raw")

    # 已解压过就跳过（节省时间）
    if os.path.isdir(raw_dir):
        n = sum(1 for _, _, fs in os.walk(raw_dir) for f in fs
                if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if n > 100:
            print("[跳过] 已经解压过 %d 张图片，无需重复下载。" % n)
            show_tree(raw_dir)
            return
    if not download(url, zip_path):
        print("\n[失败] 自动下载失败。请检查网络后重试，或手动下载：")
        print("  主源：" + DEFAULT_URL)
        print("  备用：" + BACKUP_URL)
        print("  手动下载后：把 zip 放到 %s ，再运行 prepare_dataset.py" % args.out)
        sys.exit(1)

    if zip_path.lower().endswith(".zip"):
        unzip(zip_path, raw_dir)
        show_tree(raw_dir)
    else:
        print("这是 7z 文件，请用 7-Zip 解压到：%s" % raw_dir)

    print("\n下一步：运行  py -3.12 prepare_dataset.py  划分训练/验证/测试集")


if __name__ == "__main__":
    main()
