#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict.py —— 单张图片 / 整个文件夹 的口罩佩戴识别（命令行版）

用法示例：
    py -3.12 predict.py --image demo_images\\test1.jpg
    py -3.12 predict.py --image demo_images
    py -3.12 predict.py --image C:\\Users\\me\\Pictures\\photo.png

结果会以"绿框=已佩戴口罩 / 红框=未佩戴口罩"的形式标注并保存到 outputs/predictions/
"""
import argparse
import os
import time

import numpy as np
import paddle
import paddle.nn as nn
from PIL import Image, ImageDraw

from dataset import preprocess_single_image
from model import build_model
from utils import (CLASS_CN, CLASS_NAMES, IMG_SIZE, MODEL_PATH, OUT_DIR, TEST_DIR,
                   DEMO_IMG_DIR, ensure_dirs, get_pil_font, list_images, pick_device)


def load_model(model_path=MODEL_PATH, device="auto"):
    """加载训练好的模型（推理模式）"""
    device = pick_device(device)
    paddle.set_device(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError("找不到模型文件 %s，请先运行 train.py" % model_path)
    model = build_model(num_classes=len(CLASS_CN), pretrained=False, verbose=False)
    model.set_state_dict(paddle.load(model_path))
    model.eval()
    return model, device


@paddle.no_grad()
def predict_image(model, pil_img, size=IMG_SIZE):
    """对一张 PIL 图片识别，返回 (类别索引, 置信度, 两类概率, 耗时毫秒)"""
    img = pil_img.convert("RGB")
    t0 = time.perf_counter()
    logits = model(preprocess_single_image(img, size))      # [1,3,H,W] -> [1,2]
    prob = nn.functional.softmax(logits, axis=1).numpy()[0]
    cost = (time.perf_counter() - t0) * 1000.0
    idx = int(np.argmax(prob))
    return idx, float(prob[idx]), prob, cost


def annotate(pil_img, idx, conf):
    """在图上画中文预测结果：绿框=已佩戴口罩，红框=未佩戴口罩"""
    img = pil_img.convert("RGB").copy()
    color = (0, 170, 60) if idx == 0 else (220, 30, 30)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    border = max(4, int(min(w, h) * 0.012))
    draw.rectangle([0, 0, w - 1, h - 1], outline=color, width=border)
    text = "%s  %.1f%%" % (CLASS_CN[idx], conf * 100)
    font = get_pil_font(max(18, int(min(w, h) * 0.06)))
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = max(6, int(min(w, h) * 0.018))
    draw.rectangle([border, border, border + tw + pad * 2, border + th + pad * 2], fill=color)
    draw.text((border + pad, border + pad - box[1]), text, font=font, fill=(255, 255, 255))
    return img


def resolve_images(target):
    """根据输入决定要识别哪些图片"""
    if target and os.path.isdir(target):
        return list_images(target)
    if target and os.path.isfile(target):
        return [target]
    cand = list_images(DEMO_IMG_DIR)                     # 默认找 demo_images 里的图片
    if cand:
        return cand
    print("[提示] demo_images 里没有图片，改用测试集样图演示。")
    out = []
    for cls in CLASS_NAMES:
        out += list_images(os.path.join(TEST_DIR, cls))[:4]
    return out


def main():
    parser = argparse.ArgumentParser(description="VGG16 口罩识别 —— 图片识别")
    parser.add_argument("--image", default=None, help="图片文件或文件夹路径")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--save-dir", default=os.path.join(OUT_DIR, "predictions"))
    args = parser.parse_args()
    ensure_dirs()
    os.makedirs(args.save_dir, exist_ok=True)

    model, device = load_model(args.model, args.device)
    print("=" * 68)
    print("模型：%s   运行设备：%s" % (args.model, device.upper()))

    imgs = resolve_images(args.image)
    if not imgs:
        print("[错误] 没有找到可识别的图片，请用 --image 指定图片路径")
        return
    print("待识别图片：%d 张" % len(imgs))
    print("-" * 68)

    stats = {0: 0, 1: 0}
    costs = []
    for i, p in enumerate(imgs, 1):
        try:
            pil = Image.open(p)
        except Exception as e:
            print("[跳过] %s（无法读取：%s）" % (p, e))
            continue
        idx, conf, prob, cost = predict_image(model, pil)
        stats[idx] += 1
        costs.append(cost)
        print("[%2d/%2d] %-38s -> %s  置信度 %.2f%%  (已佩戴 %.2f%% / 未佩戴 %.2f%%)  %.1f ms"
              % (i, len(imgs), os.path.basename(p)[:38], CLASS_CN[idx], conf * 100,
                 prob[0] * 100, prob[1] * 100, cost))
        out_path = os.path.join(args.save_dir,
                                "%03d_%s_%s.jpg" % (i, CLASS_NAMES[idx],
                                                    os.path.splitext(os.path.basename(p))[0][:28]))
        annotate(pil, idx, conf).save(out_path)

    print("-" * 68)
    if costs:
        print("识别完成：已佩戴口罩 %d 张，未佩戴口罩 %d 张，平均单张耗时 %.1f ms"
              % (stats[0], stats[1], sum(costs) / len(costs)))
    print("标注结果已保存到：%s" % args.save_dir)


if __name__ == "__main__":
    main()
