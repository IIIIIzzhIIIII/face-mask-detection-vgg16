#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfcheck.py —— 环境与代码自检（装好依赖后先跑这个，1 分钟内出结果）

检查内容：飞桨版本与 GPU、学习率调度器行为、VGG16 前向、数据预处理、
          OpenCV 人脸检测、绘图中文、matplotlib/scikit-learn/gradio、全部模块能否导入

用法：
    py -3.12 selfcheck.py
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

RESULTS = []


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check(name, fn):
    t0 = time.time()
    try:
        info = fn() or ""
        RESULTS.append((name, True, info))
        print("[通过] %-26s %s" % (name, str(info)[:100]))
    except Exception as e:
        msg = "%s: %s" % (type(e).__name__, e)
        RESULTS.append((name, False, msg))
        print("[失败] %-26s %s" % (name, msg))


def t_paddle():
    import paddle
    from utils import cuda_usable
    version = paddle.__version__
    if not paddle.device.is_compiled_with_cuda():
        return "版本 %s | 非 GPU 版（将使用 CPU）" % version
    try:
        num = paddle.device.cuda.device_count()
    except Exception as e:
        return ("版本 %s | 已编译 GPU 但不可用：%s | 请运行 py -3.12 fix_cudnn.py（缺 cuDNN）"
                % (version, str(e)[:50]))
    if num <= 0:
        return "版本 %s | 未检测到 GPU 设备（将使用 CPU）" % version
    if not cuda_usable():
        return ("版本 %s | 检测到 GPU 但无法使用（通常是缺 cuDNN，运行 py -3.12 fix_cudnn.py 修复）"
                "→ 本次将使用 CPU" % version)
    try:
        prop = paddle.device.cuda.get_device_properties(0)
        return "版本 %s | GPU 可用 | %s %.1fGB | 设备数 %d" % (
            version, prop.name, prop.total_memory / 1024 ** 3, num)
    except Exception:
        return "版本 %s | GPU 可用 | 设备数 %d" % (version, num)


def t_scheduler():
    """确认学习率调度器是否会被 optimizer.step() 自动推进（决定训练循环写法）"""
    import paddle
    from paddle.optimizer.lr import CosineAnnealingDecay
    param = paddle.create_parameter([2, 2], "float32", "w")
    sch = CosineAnnealingDecay(learning_rate=0.1, T_max=10)
    opt = paddle.optimizer.Momentum(learning_rate=sch, momentum=0.9, parameters=[param])
    before = int(getattr(sch, "last_epoch", -1))
    opt.step()
    after = int(getattr(sch, "last_epoch", -1))
    return "自动步进=%s (last_epoch %d -> %d)" % (after != before, before, after)


def t_transforms():
    from paddle.vision import transforms
    want = ["Resize", "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize", "Compose"]
    have = [n for n in want if hasattr(transforms, n)]
    missing = [n for n in want if n not in have]
    if missing:
        raise RuntimeError("缺少变换: %s" % missing)
    return "可用: %s" % ",".join(have)


def t_vgg_api():
    """确认飞桨官方 VGG16 接口存在（预训练权重可用）"""
    import paddle
    fn = getattr(paddle.vision.models, "vgg16", None)
    if fn is None:
        raise RuntimeError("paddle.vision.models.vgg16 不存在")
    return "paddle.vision.models.vgg16 可用"


def t_model():
    import paddle
    from model import build_model, count_parameters
    paddle.set_device("cpu")            # 前向测试固定在 CPU 上，避免 GPU 环境问题干扰判断
    m = build_model(2, pretrained=False, verbose=False)
    m.eval()
    with paddle.no_grad():
        out = m(paddle.randn([1, 3, 224, 224]))
    total, _ = count_parameters(m)
    return "输入[1,3,224,224] -> 输出%s 参数量%.2fM" % (list(out.shape), total / 1e6)


def t_dataset():
    import numpy as np
    from PIL import Image
    from dataset import build_transforms, simulate_camera
    img = Image.fromarray((np.random.rand(260, 300, 3) * 255).astype("uint8"))
    x = build_transforms(True)(img)
    y = simulate_camera(img)
    return "增强后张量%s dtype=%s | 摄像头模拟%s" % (list(x.shape), str(getattr(x, "dtype", "?")), y.size)


def t_deps():
    import cv2
    import matplotlib
    import numpy
    import sklearn
    from inference import find_haar_path
    haar = find_haar_path()
    return "cv2 %s | numpy %s | sklearn %s | matplotlib %s | 人脸检测模型 %s" % (
        cv2.__version__, numpy.__version__, sklearn.__version__, matplotlib.__version__,
        "已找到" if haar else "未找到(将退化为整图识别)")


def t_gradio():
    import gradio
    return "gradio %s" % gradio.__version__


def t_chinese():
    import numpy as np
    from utils import draw_chinese_text, find_chinese_font, setup_matplotlib_chinese
    font = find_chinese_font()
    img = np.full((80, 340, 3), 255, dtype=np.uint8)
    out = draw_chinese_text(img, [("中文测试：已佩戴口罩", (10, 20), (0, 0, 255), 24)])
    return "字体=%s | matplotlib中文=%s | 绘制OK" % (os.path.basename(font or "默认"),
                                                  setup_matplotlib_chinese())


def t_inference():
    import numpy as np
    from inference import detect_faces, find_haar_path
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    boxes = detect_faces(frame)          # 纯灰图应当检不到人脸（0 个），能跑通即可
    return "人脸检测可用（灰图上检出 %d 个，预期 0；模型=%s）" % (
        len(boxes), "有" if find_haar_path() else "无(整图识别)")


def t_modules():
    import importlib
    for m in ["utils", "dataset", "model", "predict", "inference",
              "train", "evaluate", "camera_demo", "check_camera", "prepare_dataset"]:
        importlib.import_module(m)
    return "10 个模块全部导入成功"


def main():
    try:
        import paddle
        paddle.set_device("cpu")          # 自检统一在 CPU 上运行，避免 GPU 环境问题连累其它检查项
    except Exception:
        pass
    section("1. 运行环境")
    check("飞桨 PaddlePaddle", t_paddle)
    check("学习率调度器行为", t_scheduler)
    section("2. 依赖库")
    check("OpenCV/numpy/sklearn", t_deps)
    check("Gradio 网页界面", t_gradio)
    check("图像变换 transforms", t_transforms)
    section("3. 模型与数据")
    check("官方 VGG16 接口", t_vgg_api)
    check("VGG16 前向传播", t_model)
    check("数据预处理/增强", t_dataset)
    check("中文显示与绘图", t_chinese)
    section("4. 各模块导入")
    check("inference 人脸检测", t_inference)
    check("全部脚本模块", t_modules)

    bad = [r for r in RESULTS if not r[1]]
    section("自检结果")
    print("通过 %d 项，失败 %d 项" % (len(RESULTS) - len(bad), len(bad)))
    for name, _ok, info in bad:
        print("  [失败] %s -> %s" % (name, info))
    if bad:
        print("\n请先把上面失败的项解决掉再继续训练。")
        sys.exit(1)
    print("\n环境正常，可以开始：py -3.12 download_dataset.py")


if __name__ == "__main__":
    main()
