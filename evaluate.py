#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py —— 在测试集上全面评估模型（任务书要求的全部指标都在这）

输出（outputs/ 目录）：
    metrics.json           全部指标数值（准确率、精确率、召回率、F1、mAP、AUC、混淆矩阵）
    metrics_report.txt     分类报告（可直接贴到论文里）
    confusion_matrix.png   混淆矩阵图
    roc_pr_curve.png       ROC 曲线 + PR 曲线
    misclassified.csv      误判样本清单（便于做"错误分析"）
    eval_predictions.png   测试集预览图（标注 真实/预测）

用法：
    py -3.12 evaluate.py
    py -3.12 evaluate.py --split val          # 也可在验证集上评估
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import paddle
import paddle.nn as nn

from dataset import build_eval_loader, preprocess_single_image
from model import build_model, count_parameters
from utils import (DATA_ROOT, MODEL_PATH, OUT_DIR, CLASS_NAMES, CLASS_CN, IMG_SIZE,
                   ensure_dirs, setup_matplotlib_chinese, make_grid_image, pick_device)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score, confusion_matrix,
                             classification_report, precision_recall_curve, roc_curve)


def to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


@paddle.no_grad()
def collect_predictions(model, loader):
    """遍历数据集，收集所有样本的预测概率和真实标签"""
    model.eval()
    probs, gts = [], []
    for imgs, labels in loader:
        logits = model(paddle.to_tensor(imgs))
        prob = nn.functional.softmax(logits, axis=1).numpy()
        probs.append(prob)
        gts.append(to_numpy(labels))
    return np.concatenate(probs, axis=0), np.concatenate(gts, axis=0)


@paddle.no_grad()
def measure_speed(model, device, runs=30, warmup=5):
    """测量单张图片推理耗时（论文"实时性分析"要用）"""
    model.eval()
    dummy = paddle.randn([1, 3, IMG_SIZE, IMG_SIZE])
    for _ in range(warmup):
        model(dummy)
    if device == "gpu":
        paddle.device.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(dummy)
    if device == "gpu":
        paddle.device.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0     # 毫秒/张


def plot_confusion_matrix(cm, save_path):
    setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_CN); ax.set_yticklabels(CLASS_CN)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（测试集）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(save_path, dpi=180); plt.close(fig)
    print("[OK] 混淆矩阵图：%s" % save_path)


def plot_roc_pr(y_true, y_prob, roc_auc, ap, save_path):
    setup_matplotlib_chinese()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, "b-", lw=2, label="ROC 曲线 (AUC = %.4f)" % roc_auc)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("假正率 FPR"); axes[0].set_ylabel("真正率 TPR")
    axes[0].set_title("ROC 曲线"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(rec, prec, "r-", lw=2, label="PR 曲线 (AP = %.4f)" % ap)
    axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
    axes[1].set_title("Precision-Recall 曲线"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=180); plt.close(fig)
    print("[OK] ROC/PR 曲线：%s" % save_path)


def main():
    parser = argparse.ArgumentParser(description="评估 VGG16 口罩识别模型")
    parser.add_argument("--data", default=DATA_ROOT)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    args = parser.parse_args()
    ensure_dirs()

    device = pick_device(args.device)
    paddle.set_device(device)

    if not os.path.exists(args.model):
        print("[错误] 找不到模型文件：%s，请先运行 train.py" % args.model)
        sys.exit(1)

    loader = build_eval_loader(args.data, args.split, args.batch_size, args.num_workers)
    model = build_model(num_classes=len(CLASS_CN), pretrained=False, verbose=False)
    model.set_state_dict(paddle.load(args.model))
    model.eval()
    total_p, _ = count_parameters(model)
    print("=" * 68)
    print("已加载模型：%s" % args.model)
    print("评估数据集：%s（%s 集，共 %d 张）" % (args.data, args.split, len(loader.dataset)))

    t0 = time.time()
    probs, gts = collect_predictions(model, loader)
    eval_time = time.time() - t0
    y_pred = probs.argmax(axis=1)
    y_prob = probs[:, 1]                                   # "未佩戴口罩"为正类，便于计算 AUC/AP

    # ---------------- 核心指标 ----------------
    acc = accuracy_score(gts, y_pred)
    p_macro = precision_score(gts, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(gts, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(gts, y_pred, average="macro", zero_division=0)
    p_w = precision_score(gts, y_pred, average="weighted", zero_division=0)
    r_w = recall_score(gts, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(gts, y_pred, average="weighted", zero_division=0)
    p_cls = precision_score(gts, y_pred, average=None, labels=[0, 1], zero_division=0)
    r_cls = recall_score(gts, y_pred, average=None, labels=[0, 1], zero_division=0)
    f1_cls = f1_score(gts, y_pred, average=None, labels=[0, 1], zero_division=0)
    ap_score = average_precision_score(gts, y_prob)        # 二分类平均精度 AP（即 mAP）
    auc = roc_auc_score(gts, y_prob)
    cm = confusion_matrix(gts, y_pred, labels=[0, 1])

    ms = measure_speed(model, device)
    size_mb = os.path.getsize(args.model) / 1024.0 / 1024.0

    metrics = {
        "split": args.split,
        "model_path": args.model,
        "num_samples": int(len(gts)),
        "class_names": CLASS_NAMES,
        "accuracy": float(acc),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f1_w),
        "precision_per_class": {CLASS_NAMES[i]: float(p_cls[i]) for i in range(2)},
        "recall_per_class": {CLASS_NAMES[i]: float(r_cls[i]) for i in range(2)},
        "f1_per_class": {CLASS_NAMES[i]: float(f1_cls[i]) for i in range(2)},
        "mAP_AP": float(ap_score),
        "AUC": float(auc),
        "confusion_matrix": cm.tolist(),
        "params_M": round(total_p / 1e6, 2),
        "model_size_MB": round(size_mb, 2),
        "inference_ms_per_image": round(ms, 2),
        "inference_fps": round(1000.0 / ms, 1),
        "device": device,
        "eval_time_sec": round(eval_time, 2),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---------------- 分类报告（可直接贴进论文） ----------------
    report = classification_report(gts, y_pred, labels=[0, 1], target_names=CLASS_CN,
                                   digits=4, zero_division=0)
    with open(os.path.join(OUT_DIR, "metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write("VGG16 口罩识别模型 —— %s 集评估报告\n" % args.split)
        f.write("模型：%s   设备：%s   样本数：%d\n" % (args.model, device, len(gts)))
        f.write("=" * 60 + "\n")
        f.write(report + "\n")
        f.write("准确率 Accuracy           : %.4f\n" % acc)
        f.write("精确率 Precision (macro)  : %.4f\n" % p_macro)
        f.write("召回率 Recall    (macro)  : %.4f\n" % r_macro)
        f.write("F1 值  F1-Score (macro)   : %.4f\n" % f1_macro)
        f.write("F1 值  F1-Score (weighted): %.4f\n" % f1_w)
        f.write("平均精度 mAP (AP)         : %.4f\n" % ap_score)
        f.write("AUC                       : %.4f\n" % auc)
        f.write("混淆矩阵 [[TN,FP],[FN,TP]]: %s\n" % cm.tolist())
        f.write("模型参数量                : %.2f M\n" % (total_p / 1e6))
        f.write("模型文件大小              : %.2f MB\n" % size_mb)
        f.write("单张推理耗时              : %.2f ms (%.1f FPS)\n" % (ms, 1000.0 / ms))
    print("[OK] 指标文件：outputs/metrics.json 与 outputs/metrics_report.txt")

    plot_confusion_matrix(cm, os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_roc_pr(gts, y_prob, auc, ap_score, os.path.join(OUT_DIR, "roc_pr_curve.png"))

    # ---------------- 误判样本清单（做错误分析用） ----------------
    samples = loader.dataset.samples
    bad = np.where(y_pred != gts)[0]
    with open(os.path.join(OUT_DIR, "misclassified.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["图片路径", "真实类别", "预测类别", "预测置信度", "未戴口罩概率"])
        for idx in bad:
            path = samples[idx][0] if idx < len(samples) else ""
            w.writerow([path, CLASS_CN[gts[idx]], CLASS_CN[y_pred[idx]],
                        "%.4f" % probs[idx].max(), "%.4f" % y_prob[idx]])
    print("误判样本 %d 张，明细见 outputs/misclassified.csv" % len(bad))

    # ---------------- 测试集预测预览图 ----------------
    n_show = min(12, len(samples))
    paths = [samples[i][0] for i in range(n_show)]
    labels = ["真:%s 预:%s" % ("已戴" if gts[i] == 0 else "未戴",
                              "已戴" if y_pred[i] == 0 else "未戴") for i in range(n_show)]
    make_grid_image(paths, labels, cols=4, cell=200).save(
        os.path.join(OUT_DIR, "eval_predictions.png"))
    print("[OK] 测试集预测预览图：outputs/eval_predictions.png")

    # ---------------- 控制台汇总 ----------------
    print("=" * 68)
    print("%s 集评估结果（%d 张，耗时 %.1fs）" % (args.split, len(gts), eval_time))
    print("-" * 68)
    print("  准确率 Accuracy            : %.4f" % acc)
    print("  精确率 Precision  (macro)  : %.4f" % p_macro)
    print("  召回率 Recall     (macro)  : %.4f" % r_macro)
    print("  F1 值  F1-Score   (macro)  : %.4f" % f1_macro)
    print("  F1 值  F1-Score   (weighted): %.4f" % f1_w)
    print("  平均精度 mAP (AP)          : %.4f" % ap_score)
    print("  AUC                        : %.4f" % auc)
    print("  混淆矩阵 [[TN,FP],[FN,TP]] : %s" % cm.tolist())
    print("  模型参数量 / 大小          : %.2f M / %.2f MB" % (total_p / 1e6, size_mb))
    print("  单张推理耗时               : %.2f ms  (%.1f FPS)" % (ms, 1000.0 / ms))
    print("-" * 68)
    print("下一步：py -3.12 predict.py            （单张图片识别）")
    print("        py -3.12 camera_demo.py        （实时摄像头/视频演示）")


if __name__ == "__main__":
    main()
