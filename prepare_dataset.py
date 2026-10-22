#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_dataset.py —— 整理数据集并按 8:1:1 划分 train / val / test

做三件事：
  1. 扫描 dataset/raw/ 下所有图片，按"文件夹名"自动判断类别：
         WithMask / with_mask        -> 0  已佩戴口罩
         WithoutMask / without_mask  -> 1  未佩戴口罩
         Mask_Social_Distancing      -> 丢弃（不属于本次二分类任务）
  2. 每类随机抽样（默认每类最多 3000 张，控制训练时间），按 8:1:1 分层划分
  3. 拷贝到 dataset/facemask/{train,val,test}/{with_mask,without_mask}/
     并生成 outputs/dataset_samples.png（12 张样图拼图）用于肉眼确认数据正确

用法：
    py -3.12 prepare_dataset.py
    py -3.12 prepare_dataset.py --max-per-class 0     # 0 = 使用全部图片
    py -3.12 prepare_dataset.py --seed 42 --force
"""
import argparse
import os
import random
import shutil
import sys

import numpy as np

from utils import (BASE_DIR, DATASET_DIR, DATA_ROOT, CLASS_NAMES, CLASS_CN,
                   OUT_DIR, ensure_dirs, make_grid_image, list_images, count_samples)

RAW_DIR = os.path.join(DATASET_DIR, "raw")
MIN_BYTES = 2 * 1024       # 小于 2KB 的图片视为损坏，直接丢弃

# 需要丢弃的文件夹关键字（第三类：口罩佩戴不规范/社交距离，不属于本次二分类任务）
DROP_KEYS = ["socialdistancing", "masksocialdistancing", "incorrect", "wrongmask"]


def norm_key(name):
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


def classify_folder(folder_name):
    """
    根据文件夹名判断类别：
        含 without / no mask  -> 1 未佩戴口罩
        含 with / mask        -> 0 已佩戴口罩
        第三类（社交距离等）   -> None（丢弃）
    """
    f = norm_key(folder_name)
    if any(k in f for k in DROP_KEYS):
        return None
    if "without" in f or f in ("nomask", "no_mask", "0", "unmasked"):
        return 1
    if "with" in f or f in ("mask", "1", "masked"):
        return 0
    return None


def scan_raw(raw_dir):
    """扫描原始目录，返回 {0: [路径...], 1: [路径...]} 与丢弃统计"""
    buckets = {0: [], 1: []}
    dropped = 0
    folders = {}
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        folder = os.path.basename(dirpath)
        imgs = [f for f in filenames
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        if not imgs:
            continue
        cls = classify_folder(folder)
        if cls is None:
            if any(k in norm_key(folder) for k in DROP_KEYS):
                dropped += len(imgs)
            continue
        folders[folder] = folders.get(folder, 0) + len(imgs)
        for f in imgs:
            p = os.path.join(dirpath, f)
            try:
                if os.path.getsize(p) >= MIN_BYTES:
                    buckets[cls].append(p)
            except OSError:
                pass
    if folders:
        print("  识别到的文件夹（文件夹名 -> 图片数）：")
        for name, n in sorted(folders.items()):
            print("      %-28s %d 张" % (name, n))
    return buckets, dropped


def split_list(items, seed, ratios=(0.8, 0.1, 0.1)):
    rnd = random.Random(seed)
    items = list(items)
    rnd.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def copy_files(paths, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for i, src in enumerate(paths):
        ext = os.path.splitext(src)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            ext = ".jpg"
        shutil.copy2(src, os.path.join(dst_dir, "%06d%s" % (i, ext)))
    return len(paths)


def main():
    ap = argparse.ArgumentParser(description="整理并划分口罩数据集")
    ap.add_argument("--raw", default=RAW_DIR, help="原始数据目录（download_dataset.py 解压出来的）")
    ap.add_argument("--out", default=DATA_ROOT, help="划分后的输出目录")
    ap.add_argument("--max-per-class", type=int, default=3000,
                    help="每类最多使用多少张（0=全部；默认为控制训练时间取 3000）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="已存在时强制重新划分")
    args = ap.parse_args()
    ensure_dirs()

    if not os.path.isdir(args.raw):
        print("[错误] 找不到原始数据目录：%s" % args.raw)
        print("       请先运行：py -3.12 download_dataset.py")
        sys.exit(1)

    exist = count_samples(args.out)
    if sum(exist.values()) > 0 and not args.force:
        print("[跳过] 数据集已划分完成（--force 可强制重做）：")
        for k, v in exist.items():
            print("    %-26s %d 张" % (k, v))
        return

    print("正在扫描原始图片：%s" % args.raw)
    buckets, dropped = scan_raw(args.raw)
    print("  检出 第1类(已佩戴) %d 张，第2类(未佩戴) %d 张，丢弃(第三类) %d 张"
          % (len(buckets[0]), len(buckets[1]), dropped))
    if len(buckets[0]) == 0 or len(buckets[1]) == 0:
        print("[错误] 某一类没有找到图片，请检查 %s 的目录结构（应为 WithMask/WithoutMask 之类）" % args.raw)
        sys.exit(1)

    # 每类抽样
    rnd = random.Random(args.seed)
    for c in [0, 1]:
        rnd.shuffle(buckets[c])
        if args.max_per_class and len(buckets[c]) > args.max_per_class:
            buckets[c] = buckets[c][:args.max_per_class]

    sample_paths, sample_labels = [], []
    for c in [0, 1]:
        tr, va, te = split_list(buckets[c], args.seed)
        print("\n类别 [%s]：共 %d 张  -> train %d / val %d / test %d"
              % (CLASS_CN[c], len(buckets[c]), len(tr), len(va), len(te)))
        for split, files in [("train", tr), ("val", va), ("test", te)]:
            n = copy_files(files, os.path.join(args.out, split, CLASS_NAMES[c]))
            print("    %-6s/%-13s %d 张" % (split, CLASS_NAMES[c], n))
        # 每类抽 6 张做样图
        for p in tr[:6]:
            sample_paths.append(p)
            sample_labels.append(CLASS_CN[c])

    # 打乱样图顺序，避免前 6 张全是同一类
    order = list(range(len(sample_paths)))
    random.Random(args.seed).shuffle(order)
    sample_paths = [sample_paths[i] for i in order]
    sample_labels = [sample_labels[i] for i in order]
    grid = make_grid_image(sample_paths, sample_labels, cols=4, cell=200)
    grid_path = os.path.join(OUT_DIR, "dataset_samples.png")
    grid.save(grid_path)
    print("\n[OK] 数据集检查图已保存：%s（请打开看一眼，确认都是人脸+口罩的照片）" % grid_path)

    stat = count_samples(args.out)
    total = sum(stat.values())
    print("\n===== 划分完成，总计 %d 张 =====" % total)
    for k, v in stat.items():
        print("    %-26s %d 张" % (k, v))
    print("\n下一步：运行  py -3.12 train.py")


if __name__ == "__main__":
    main()
