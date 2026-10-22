#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model.py —— VGG16 网络定义（全项目唯一一份，训练/评估/推理/界面都用它）

网络结构（与开题报告"基于 VGG16 搭建模型"一致）：
    5 个卷积块，共 13 层卷积 + 5 次最大池化
    输入 224x224x3 -> features -> 7x7x512 -> flatten -> 4096 -> 4096 -> 2

另外提供 build_model()：
    pretrained=True  时，从飞桨官方仓库下载 ImageNet 预训练权重并进行"按名字+形状匹配"的迁移，
                     加快收敛、提升精度（论文里可作为"迁移学习优化"的小亮点）；
    freeze_features=True 时冻结卷积层只训练分类层（训练极快，适合先验证流程）。
"""
import paddle
import paddle.nn as nn


class VGG16(nn.Layer):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # block1
            nn.Conv2D(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2D(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2D(2, 2),
            # block2
            nn.Conv2D(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2D(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2D(2, 2),
            # block3
            nn.Conv2D(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2D(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2D(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2D(2, 2),
            # block4
            nn.Conv2D(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2D(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2D(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2D(2, 2),
            # block5
            nn.Conv2D(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2D(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2D(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2D(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_imagenet_weights(model, verbose=True):
    """
    把飞桨官方 VGG16（ImageNet 1000 类）的预训练权重复制到本模型。
    按"层名 + 形状"匹配：形状不同的层（最后一层 1000->2）自动跳过并随机初始化。
    返回：(成功迁移的层数, 跳过的层名列表)
    """
    try:
        # 飞桨官方 VGG16 实现导出的是小写函数 vgg16()，首次运行会自动下载 ImageNet 权重并校验
        src = paddle.vision.models.vgg16(pretrained=True)
    except Exception as e:
        if verbose:
            print("[迁移学习] 预训练权重下载/加载失败：%s: %s" % (type(e).__name__, e))
            print("[迁移学习] 本次改为从零训练（联网后可重试，或用 --scratch 明确指定从零训练）")
        return 0, []
    src_sd = src.state_dict()
    dst_sd = model.state_dict()
    new_sd, loaded, skipped = {}, 0, []
    for k, v in dst_sd.items():
        if k in src_sd and list(src_sd[k].shape) == list(v.shape):
            new_sd[k] = src_sd[k]
            loaded += 1
        else:
            new_sd[k] = v
            skipped.append(k)
    model.set_state_dict(new_sd)
    del src
    if verbose:
        print("[迁移学习] 从 ImageNet 预训练权重载入 %d 个参数层" % loaded)
        if skipped:
            print("[迁移学习] 未匹配(随机初始化)的层：%s" % ", ".join(skipped))
    return loaded, skipped


def freeze_features(model):
    """冻结卷积特征提取层（只训练最后的全连接分类层）"""
    for p in model.features.parameters():
        p.stop_gradient = True
    return model


def count_parameters(model):
    """统计模型参数量（论文"模型分析"章节可用）"""
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if not p.stop_gradient)
    return total, trainable


def build_model(num_classes=2, pretrained=False, freeze_features_flag=False, verbose=True):
    model = VGG16(num_classes=num_classes)
    model.pretrained_loaded = False
    if pretrained:
        loaded, _ = load_imagenet_weights(model, verbose=verbose)
        model.pretrained_loaded = loaded > 0
    if freeze_features_flag:
        freeze_features(model)
        if verbose:
            print("[迁移学习] 已冻结卷积层，仅训练分类层")
    return model
