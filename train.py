#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py —— VGG16 口罩识别模型训练脚本

与开题报告一致的配置：
    * 模型：VGG16（13 层卷积 + 3 层全连接）
    * 优化器：SGD，momentum=0.9，学习率余弦退火(CosineAnnealingDecay)
    * 损失函数：CrossEntropyLoss（输出经 softmax）
    * 训练轮数：默认 10 轮（迁移学习微调）；--scratch 时从零训练建议 30 轮以上

用法：
    py -3.12 train.py                                # 推荐：ImageNet 预训练微调，10 轮
    py -3.12 train.py --scratch --epochs 30           # 完全从零训练（严格对应开题报告）
    py -3.12 train.py --freeze --epochs 5             # 冻结卷积层，只训练分类层（极快）
    py -3.12 train.py --epochs 20 --lr 0.0005         # 自定义

输出（全部在 outputs/ 下）：
    best_model.pdparams      验证集最优模型
    last_model.pdparams      最后一轮模型
    train_log.csv            每轮 loss/acc 日志
    loss_acc_curve.png       损失/准确率曲线图（论文插图）
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import paddle
import paddle.nn as nn
from paddle.optimizer.lr import CosineAnnealingDecay

from dataset import build_loaders
from model import build_model, count_parameters
from utils import (DATA_ROOT, MODEL_PATH, OUT_DIR, ensure_dirs, CLASS_CN,
                   pick_device, setup_matplotlib_chinese)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def scheduler_auto_stepped():
    """
    探测当前飞桨版本是否会在 optimizer.step() 里自动推进学习率调度器。
    （不同版本行为不一致：自动推进时若再手动 step 会导致学习率衰减过快，因此这里做自适应）
    """
    try:
        param = paddle.create_parameter([2], "float32", "probe_w")
        sch = CosineAnnealingDecay(learning_rate=0.1, T_max=10)
        opt = paddle.optimizer.Momentum(learning_rate=sch, momentum=0.9, parameters=[param])
        before = int(getattr(sch, "last_epoch", -1))
        opt.step()
        return int(getattr(sch, "last_epoch", -1)) != before
    except Exception:
        return False


def build_optimizer(parameters, learning_rate, momentum=0.9, weight_decay=5e-4):
    """
    构造"带动量的随机梯度下降"优化器。
    注意：飞桨里 paddle.optimizer.SGD 不支持 momentum 参数，
          带动量的 SGD 对应 paddle.optimizer.Momentum（公式即 SGD + 动量项）。
    """
    try:
        return paddle.optimizer.Momentum(learning_rate=learning_rate, momentum=momentum,
                                         parameters=parameters, weight_decay=weight_decay)
    except TypeError:
        return paddle.optimizer.SGD(learning_rate=learning_rate, momentum=momentum,
                                    parameters=parameters, weight_decay=weight_decay)


def get_lr_safe(optimizer, scheduler):
    """安全获取当前学习率（兼容不同飞桨版本的接口差异）"""
    try:
        return float(optimizer.get_lr())
    except Exception:
        pass
    for attr in ("last_lr", "get_lr"):
        try:
            v = getattr(scheduler, attr)
            return float(v() if callable(v) else v)
        except Exception:
            continue
    return float("nan")


def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, correct, total, steps = 0.0, 0, 0, 0
    for step, (imgs, labels) in enumerate(loader):
        imgs = paddle.to_tensor(imgs)
        labels = paddle.to_tensor(labels, dtype="int64")
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        total_loss += float(loss.numpy())
        correct += int(paddle.sum(paddle.argmax(logits, axis=1) == labels).numpy())
        total += int(labels.shape[0])
        steps = step + 1
    return total_loss / max(steps, 1), correct / max(total, 1)


@paddle.no_grad()
def evaluate_split(model, loader, loss_fn, desc=""):
    model.eval()
    total_loss, correct, total, steps = 0.0, 0, 0, 0
    for step, (imgs, labels) in enumerate(loader):
        imgs = paddle.to_tensor(imgs)
        labels = paddle.to_tensor(labels, dtype="int64")
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        total_loss += float(loss.numpy())
        correct += int(paddle.sum(paddle.argmax(logits, axis=1) == labels).numpy())
        total += int(labels.shape[0])
        steps = step + 1
    loss, acc = total_loss / max(steps, 1), correct / max(total, 1)
    if desc:
        print("    [%s] loss=%.4f  acc=%.4f" % (desc, loss, acc))
    return loss, acc


def plot_curves(history, save_path):
    setup_matplotlib_chinese()
    ep = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ep, history["train_loss"], "o-", label="训练集损失")
    axes[0].plot(ep, history["val_loss"], "s-", label="验证集损失")
    axes[0].set_xlabel("训练轮数 (epoch)"); axes[0].set_ylabel("损失值 (Loss)")
    axes[0].set_title("损失曲线"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(ep, history["train_acc"], "o-", label="训练集准确率")
    axes[1].plot(ep, history["val_acc"], "s-", label="验证集准确率")
    axes[1].set_xlabel("训练轮数 (epoch)"); axes[1].set_ylabel("准确率 (Accuracy)")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("准确率曲线"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle("VGG16 口罩识别模型训练过程", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print("[OK] 曲线图已保存：%s" % save_path)


def main():
    ap = argparse.ArgumentParser(description="训练 VGG16 口罩识别模型")
    ap.add_argument("--data", default=DATA_ROOT, help="数据集根目录（含 train/val/test）")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="学习率（默认：预训练 0.001 / 从零 0.01）")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--scratch", action="store_true", help="不使用 ImageNet 预训练权重，完全从零训练")
    ap.add_argument("--freeze", action="store_true", help="冻结卷积层，只训练最后的分类层")
    ap.add_argument("--no-aug", dest="aug", action="store_false", default=True,
                    help="关闭模拟摄像头的画质增强（默认开启）")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    args = ap.parse_args()
    ensure_dirs()
    set_seed(args.seed)

    # ---------------- 运行设备 ----------------
    device = pick_device(args.device)
    paddle.set_device(device)
    print("=" * 68)
    print("运行设备：%s" % device.upper())
    if device == "gpu":
        try:
            prop = paddle.device.cuda.get_device_properties(0)
            print("GPU 型号：%s" % prop.name)
            print("显存总量：%.2f GB" % (prop.total_memory / 1024 ** 3))
        except Exception as e:
            print("(GPU 信息读取失败，忽略：%s)" % e)
    print("飞桨版本：%s" % paddle.__version__)

    # ---------------- 数据 ----------------
    print("数据集目录：%s" % args.data)
    train_loader, val_loader, test_loader = build_loaders(
        args.data, batch_size=args.batch_size, num_workers=args.num_workers, train_aug=args.aug)
    print("样本数量：训练 %d / 验证 %d / 测试 %d" %
          (len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
    if len(train_loader.dataset) == 0:
        print("[错误] 训练集为空，请先运行 prepare_dataset.py")
        sys.exit(1)

    # ---------------- 模型 ----------------
    pretrained = not args.scratch
    model = build_model(num_classes=len(CLASS_CN), pretrained=pretrained,
                        freeze_features_flag=args.freeze)
    total_p, trainable_p = count_parameters(model)
    pretrained_loaded = bool(getattr(model, "pretrained_loaded", False))
    if pretrained and not pretrained_loaded:
        print("[提示] 未能加载 ImageNet 预训练权重，本次按从零训练设置学习率")
    print("模型参数量：%.2f M（可训练 %.2f M）" % (total_p / 1e6, trainable_p / 1e6))

    lr = args.lr if args.lr is not None else (0.001 if pretrained_loaded else 0.01)
    loss_fn = nn.CrossEntropyLoss()
    # 余弦退火学习率：自动推进时按"迭代次数"退火，否则按"轮数"退火（两种都正确）
    lr_auto = scheduler_auto_stepped()
    t_max = max(args.epochs * len(train_loader), 1) if lr_auto else max(args.epochs, 1)
    scheduler = CosineAnnealingDecay(learning_rate=lr, T_max=t_max, eta_min=lr * 0.01)
    optimizer = build_optimizer(model.parameters(), scheduler, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    print("优化器：SGD+动量（paddle.optimizer.Momentum）  lr=%.5f  momentum=%.2f  weight_decay=%.1e"
          % (lr, args.momentum, args.weight_decay))
    print("学习率策略：CosineAnnealingDecay(T_max=%d, 自动步进=%s)" % (t_max, lr_auto))
    print("=" * 68)

    history = {"epoch": [], "lr": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch = -1.0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_acc = evaluate_split(model, val_loader, loss_fn)
        if not lr_auto:                 # 调度器不被 optimizer 自动推进时，手动按轮推进
            scheduler.step()
        cur_lr = get_lr_safe(optimizer, scheduler)

        history["epoch"].append(epoch); history["lr"].append(cur_lr)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        print("Epoch %2d/%d | 训练 loss=%.4f acc=%.4f | 验证 loss=%.4f acc=%.4f | lr=%.2e | %.1fs"
              % (epoch, args.epochs, tr_loss, tr_acc, va_loss, va_acc, cur_lr, time.time() - ep_t0))

        # 保存验证集最优模型（先比准确率，准确率相同则比损失）
        if va_acc > best_acc or (va_acc == best_acc and va_loss <= min(history["val_loss"])):
            best_acc, best_epoch = va_acc, epoch
            paddle.save(model.state_dict(), MODEL_PATH)
            print("    -> 验证集表现提升，已保存最优模型：%s" % MODEL_PATH)

    paddle.save(model.state_dict(), os.path.join(OUT_DIR, "last_model.pdparams"))

    # ---------------- 日志与曲线 ----------------
    with open(os.path.join(OUT_DIR, "train_log.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(history["epoch"])):
            w.writerow([history["epoch"][i], "%.6f" % history["lr"][i],
                        "%.6f" % history["train_loss"][i], "%.6f" % history["train_acc"][i],
                        "%.6f" % history["val_loss"][i], "%.6f" % history["val_acc"][i]])
    with open(os.path.join(OUT_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    plot_curves(history, os.path.join(OUT_DIR, "loss_acc_curve.png"))

    print("=" * 68)
    print("训练完成：共 %d 轮，耗时 %.1f 分钟" % (args.epochs, (time.time() - t0) / 60))
    print("验证集最高准确率：%.4f（第 %d 轮）" % (best_acc, best_epoch))
    print("最优模型：%s" % MODEL_PATH)
    print("下一步：py -3.12 evaluate.py    （在测试集上计算全部指标）")


if __name__ == "__main__":
    main()
