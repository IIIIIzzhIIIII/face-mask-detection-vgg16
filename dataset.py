#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset.py —— 数据集类与数据预处理（训练/验证/测试 + 推理 全部共用这里）

预处理流程（对应任务书要求的"缩放、裁剪、归一化"）：
    训练集：Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip
            -> 模拟摄像头扰动(亮度/对比度/噪声/模糊)   ← 提升真实摄像头下的鲁棒性
            -> ToTensor -> Normalize(ImageNet 均值方差)
    验证/测试：Resize(224) -> ToTensor -> Normalize

说明：为了消除 Paddle 版本差异带来的风险，这里的几何变换用 paddle.vision.transforms，
      "模拟摄像头扰动"用 PIL/numpy 自己实现（代码透明、可读、论文里好写）。
"""
import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import paddle
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms

from utils import CLASS_NAMES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, list_images


# ---------------------------------------------------------------- 数据增强
def _bilinear():
    """取双线性插值常量（Pillow 新版把常量移到了 Image.Resampling 下）"""
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def simulate_camera(img):
    """
    模拟摄像头/手机拍摄时的画质退化：随机亮度、对比度、高斯噪声、轻微模糊。
    目的：让模型在真实摄像头画面下也能稳定识别（而不是只在数据集上有效）。
    """
    # 亮度 / 对比度
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    # 轻微模糊（对焦不准 / 运动模糊）
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.6)))
    # 高斯噪声（摄像头暗光噪点）
    if random.random() < 0.4:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    # 随机小角度旋转（拍摄角度不固定）
    if random.random() < 0.3:
        img = img.rotate(random.uniform(-8, 8), resample=_bilinear(), expand=False)
    return img


# ---------------------------------------------------------------- 预处理
def build_transforms(train=True):
    """返回 paddle 的 Compose 预处理流水线"""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),            # 缩放
            transforms.RandomCrop(IMG_SIZE),          # 随机裁剪（数据增强）
            transforms.RandomHorizontalFlip(),        # 随机水平翻转（数据增强）
            transforms.ToTensor(),                    # HWC uint8 -> CHW float32 [0,1]
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),   # 归一化
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------- 数据集类
class MaskDataset(Dataset):
    """
    目录结构：
        root/train/with_mask/*.jpg
        root/train/without_mask/*.jpg
        root/val/...  root/test/...
    """
    def __init__(self, root, split="train", transform=None, camera_aug=False):
        super().__init__()
        self.root = os.path.join(root, split)
        self.transform = transform
        self.camera_aug = camera_aug          # 仅训练集开启"模拟摄像头"增强
        self.samples = []                     # [(图片路径, 标签), ...]
        for label, cls in enumerate(CLASS_NAMES):     # with_mask=0, without_mask=1
            cls_dir = os.path.join(self.root, cls)
            for p in list_images(cls_dir):
                self.samples.append((p, label))
        if len(self.samples) == 0:
            raise FileNotFoundError(
                "在 %s 下没有找到任何图片，请先运行 prepare_dataset.py" % self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # 极少数损坏图片：用灰图兜底，避免训练中断
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.camera_aug:
            img = simulate_camera(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, np.int64(label)


def build_loaders(data_root, batch_size=32, num_workers=0, train_aug=False):
    """构建 train / val / test 三个 DataLoader"""
    train_ds = MaskDataset(data_root, "train", build_transforms(True), camera_aug=train_aug)
    val_ds = MaskDataset(data_root, "val", build_transforms(False))
    test_ds = MaskDataset(data_root, "test", build_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader


def build_eval_loader(data_root, split="test", batch_size=32, num_workers=0):
    ds = MaskDataset(data_root, split, build_transforms(False))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def preprocess_single_image(pil_img, size=IMG_SIZE):
    """
    推理专用：把一张 PIL 图片变成模型输入 tensor（形状 [1,3,H,W]）
    与验证集预处理完全一致，保证训练/推理一致。
    """
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = tf(pil_img.convert("RGB"))
    return paddle.unsqueeze(tensor, 0)
