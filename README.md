# 基于 PaddlePaddle + VGG16 的口罩佩戴检测系统

> 本科毕业设计项目 · Windows + Python 3.12 + PaddlePaddle GPU
>
> 使用 VGG16 对「已佩戴口罩 / 未佩戴口罩」进行二分类，结合 OpenCV Haar 人脸检测、Gradio Web 界面与实时摄像头演示，覆盖数据下载、训练、评估、推理和部署全流程。

## 一、项目简介

本项目面向公共场所的口罩佩戴检测场景，输入一张图片、摄像头画面或视频，先检测画面中的人脸，再逐张判断是否佩戴口罩，并在画面上用中文标注结果：

- **绿色**：已佩戴口罩
- **红色**：未佩戴口罩
- **橙色**：置信度低于阈值，无法确定

核心工作包括：

1. 基于 PaddlePaddle 从零搭建 **VGG16**（13 层卷积 + 3 层全连接，约 1.34 亿参数）；
2. 支持 ImageNet 预训练权重迁移学习，也支持 `--scratch` 完全从零训练；
3. 训练阶段加入翻转、随机裁剪、亮度/对比度/噪声/模糊等数据增强，并模拟摄像头画质退化；
4. 使用 OpenCV Haar 级联检测人脸，支持单帧多张人脸；
5. 提供命令行图片识别、Gradio Web 界面、实时摄像头/视频/屏幕/图片流演示五种使用方式；
6. 在测试集上输出准确率、精确率、召回率、F1、mAP、AUC、混淆矩阵、ROC/PR 曲线和误判样本清单。

## 二、实验结果

实验在 600 张测试集图片（已佩戴口罩 300 张、未佩戴口罩 300 张）上评估：

| 指标 | 数值 |
|---|---|
| Accuracy（准确率） | **1.0000** |
| Precision（macro，精确率） | **1.0000** |
| Recall（macro，召回率） | **1.0000** |
| F1-Score（macro） | **1.0000** |
| mAP（AP） | **1.0000** |
| AUC | **1.0000** |
| 混淆矩阵 `[[TN, FP], [FN, TP]]` | `[[300, 0], [0, 300]]` |
| 模型参数量 / 文件大小 | 134.27 M / 512.20 MB |
| 单张推理耗时（GPU） | 8.26 ms（约 121 FPS） |
| 测试集评估耗时 | 13.62 s |

训练配置：ImageNet 预训练迁移学习，SGD + Momentum 0.9，CosineAnnealingDecay，8 个 epoch；验证集准确率在第 1 轮即达到 100%。

> 说明：该数据集类别特征较明显，因此指标较高。以上数值来自本项目实际运行日志，完整数据见 `outputs/metrics.json`、`outputs/metrics_report.txt`、`outputs/train_log.csv`。

训练曲线、混淆矩阵、ROC/PR 曲线等论文插图位于 `outputs/`：

| 文件 | 内容 |
|---|---|
| `outputs/loss_acc_curve.png` | 训练损失 / 准确率曲线 |
| `outputs/confusion_matrix.png` | 混淆矩阵热力图 |
| `outputs/roc_pr_curve.png` | ROC 曲线（AUC）与 PR 曲线（AP） |
| `outputs/eval_predictions.png` | 测试集预测预览 |
| `outputs/dataset_samples.png` | 数据集样图 |

## 三、项目结构

```text
.
├── app.py                  # Gradio Web 界面（上传图片 / 拍照识别）
├── camera_demo.py          # 实时演示：摄像头 / 视频 / 图片流 / 屏幕 / demostream
├── check_camera.py         # 摄像头设备自检
├── dataset.py              # 数据集类、预处理与数据增强
├── download_dataset.py     # 自动下载口罩数据集（国内镜像、断点续传）
├── evaluate.py             # 测试集指标评估与图表生成
├── fix_cudnn.py            # Windows 下自动安装 cuDNN，让 GPU 可用
├── get_face_model.py       # 下载 OpenCV Haar 人脸检测模型
├── inference.py            # 人脸检测 + 口罩识别 + 结果绘制
├── model.py                # VGG16 网络定义与迁移学习
├── predict.py              # 单张图片 / 整个文件夹识别
├── prepare_dataset.py      # 按 8:1:1 划分 train / val / test
├── selfcheck.py            # 环境与代码一键自检
├── train.py                # 模型训练
├── utils.py                # 路径、中文显示、设备选择等公共工具
├── requirements.txt        # Python 依赖
├── run_all.bat             # 一键下载 → 划分 → 训练 → 评估
├── demo_images/            # 演示图片（仓库自带 4 张样图）
├── models/                 # OpenCV Haar 人脸检测模型（已随仓库提供）
├── outputs/                # 评估指标、图表、截图、模型权重（权重不入库）
└── dataset/                # 数据集目录（不入库，运行下载脚本生成）
```

## 四、环境要求

- Windows 10 / 11（Linux 也可运行，路径与中文字体做了兼容处理）
- Python **3.12**
- PaddlePaddle GPU **2.6.2**（没有 NVIDIA GPU 时可装 CPU 版，功能不变、训练较慢）
- OpenCV **4.10.0.84**（必须使用 4.x：OpenCV 5 已移除 Haar 人脸检测接口）
- 其他依赖见 `requirements.txt`

> 仓库不包含 `dataset/` 与 `outputs/*.pdparams`：数据集约 930MB，训练好的模型单文件 512MB，均超过 GitHub 100MB 单文件限制。按下方步骤可自动下载数据并训练生成模型。

## 五、快速开始

### 1. 克隆项目

```bat
git clone https://github.com/IIIIIzzhIIIII/face-mask-detection-vgg16.git
cd face-mask-detection-vgg16
```

> 如果你的 Python 命令不是 `py -3.12`，请按实际环境调整（如 `python`、`python3`）。

### 2. 安装依赖（只需一次）

```bat
py -3.12 -m pip install paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
py -3.12 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 可选：Windows 下缺 cuDNN 时执行一次，让 GPU 可用（约 700MB）
py -3.12 fix_cudnn.py

:: 环境自检：飞桨 / GPU / 依赖 / 模型 / 中文显示是否正常
py -3.12 selfcheck.py
```

`selfcheck.py` 全部显示「通过」即安装成功。GPU 一栏若显示「将使用 CPU」也能正常训练，只是速度较慢。

### 3. 准备数据、训练与评估

```bat
py -3.12 download_dataset.py      :: 1. 下载数据集（约 330MB，支持断点续传）
py -3.12 prepare_dataset.py       :: 2. 按 8:1:1 划分，并生成 outputs\dataset_samples.png
py -3.12 train.py                 :: 3. 训练（默认 ImageNet 预训练微调 10 轮）
py -3.12 evaluate.py              :: 4. 测试集评估，输出全部指标与图表
```

也可以直接双击 `run_all.bat`，自动依次完成上述四步。

### 4. 关于模型权重

训练产物为 `outputs/best_model.pdparams`（512MB），因超过 GitHub 单文件限制未提交到仓库。二选一即可：

- **自行训练**：按上一步运行 `train.py`，GPU 环境约 20~30 分钟；
- **已有权重**：把 `best_model.pdparams` 放到 `outputs/` 目录，所有推理脚本会自动加载。

### 5. 训练命令的几种用法

```bat
py -3.12 train.py                          :: 迁移学习微调（推荐，10 轮，约 20~30 分钟）
py -3.12 train.py --scratch --epochs 30    :: 完全从零训练（严格对应开题报告的「从零搭建 VGG16」）
py -3.12 train.py --freeze --epochs 5      :: 冻结卷积层只训练分类层（5 分钟出结果）
py -3.12 train.py --epochs 20 --lr 0.0005  :: 自定义轮数与学习率
```

## 六、数据集说明

- 数据来源：HuggingFace 国内镜像上的公开数据集（内容等价于 Kaggle Face Mask 12k 数据集），由 `download_dataset.py` 自动下载；
- 规模：约 330MB、1 万张真实口罩人脸图；
- 划分：`prepare_dataset.py` 默认每类最多抽样 3000 张，按 8:1:1 分层划分 train / val / test；
- 类别：`with_mask`（已佩戴）与 `without_mask`（未佩戴），其它文件夹（如社交距离）自动丢弃；
- 数据集不随仓库分发，请运行脚本获取，并遵守原始数据集的使用许可。

```bat
py -3.12 download_dataset.py
py -3.12 prepare_dataset.py
py -3.12 prepare_dataset.py --max-per-class 0    :: 使用全部图片（更慢、精度略高）
py -3.12 prepare_dataset.py --force              :: 重新划分
```

## 七、系统功能一览

| 功能 | 文件 | 说明 |
|---|---|---|
| 环境自检 | `selfcheck.py` | 一键检查飞桨/GPU/依赖/模型/中文显示是否正常（**装完环境先跑这个**） |
| GPU 修复 | `fix_cudnn.py` | Windows 下飞桨缺 cuDNN 时自动安装（报 cudnn64_8.dll 错误就用它） |
| 数据集自动下载 | `download_dataset.py` | 从国内镜像下载约 330MB、1 万张真实口罩人脸图，免 Kaggle 登录 |
| 数据集整理划分 | `prepare_dataset.py` | 自动按文件夹名分两类（丢弃第三类），8:1:1 划分 train/val/test |
| 数据加载与增强 | `dataset.py` | 缩放/裁剪/翻转/归一化 + 模拟摄像头画质增强 |
| 网络模型 | `model.py` | VGG16（13 卷积 + 3 全连接），支持 ImageNet 预训练迁移 |
| 模型训练 | `train.py` | SGD + momentum 0.9 + 余弦退火，CrossEntropyLoss，保存最优模型 |
| 指标评估 | `evaluate.py` | 准确率/精确率/召回率/F1(macro,weighted)/mAP(AP)/AUC + 混淆矩阵 + 分类报告 |
| 图片识别 | `predict.py` | 单张图片或整个文件夹识别，输出标注图 |
| Web 界面 | `app.py` | Gradio 网页版，上传图片或拍照识别，显示结果与置信度 |
| 摄像头自检 | `check_camera.py` | 逐个探测摄像头设备，告诉你能否使用 |
| 实时演示 | `camera_demo.py` | 摄像头/视频/图片流/屏幕/demostream 五种输入源的实时检测 |
| 一键运行 | `run_all.bat` | 双击即可完成「下载→划分→训练→评估」 |


## 八、三种推理方式（答辩演示）

```bat
:: 方式1：单张图片 / 整个文件夹
py -3.12 predict.py --image demo_images
py -3.12 predict.py --image demo_images\sample_with_mask_1.jpg

:: 方式2：Web 界面（推荐用于答辩展示，浏览器里上传图片或直接拍照）
py -3.12 app.py
:: 浏览器打开 http://127.0.0.1:7860 ，左侧上传图片 → 点「开始识别」

:: 方式3：实时检测（五种输入源，同一套流程）
py -3.12 check_camera.py                            :: 先自检摄像头
py -3.12 camera_demo.py --source demostream          :: 默认：测试集图片合成实时视频
py -3.12 camera_demo.py --source 0                   :: 用真实摄像头（修好后直接改成 0）
py -3.12 camera_demo.py --source demo_images\test.mp4 :: 用手机录的视频
py -3.12 camera_demo.py --source screen              :: 识别屏幕画面
py -3.12 camera_demo.py --source demostream --max-frames 300 --out-video auto
```

`camera_demo.py` 运行中：`Q/ESC` 退出、`S` 保存截图、`R` 开始/停止录像。
截图与录像自动保存到 `outputs\screenshots\`、`outputs\videos\`。

## 九、输出文件说明（论文素材）

| 文件 | 用途 |
|---|---|
| `outputs/loss_acc_curve.png` | 训练损失/准确率曲线（论文「训练过程」插图） |
| `outputs/train_log.csv` | 每轮 loss、acc、学习率（可做成表格） |
| `outputs/metrics_report.txt` | 分类报告：各类精确率/召回率/F1 + 全部指标 |
| `outputs/metrics.json` | 指标数值（准确率、精确率、召回率、F1、mAP、AUC、参数量、推理耗时） |
| `outputs/confusion_matrix.png` | 混淆矩阵热力图 |
| `outputs/roc_pr_curve.png` | ROC 曲线（AUC）与 PR 曲线（AP） |
| `outputs/misclassified.csv` | 误判样本清单（论文「误差分析」章节） |
| `outputs/dataset_samples.png` | 数据集样图（确认数据正确、论文「数据集介绍」插图） |
| `outputs/screenshots/` | Web 界面与实时检测的截图（运行时生成） |
| `outputs/videos/` | 实时检测演示录像（运行时生成） |

## 十、识别结果的显示约定

自动生成的图像/画面上会标注中文：**绿色 = 已佩戴口罩，红色 = 未佩戴口罩，橙色 = 置信度太低无法确定**。

## 十一、常见问题

1. **显存不足 / 训练报 OOM**：把 `--batch-size` 调小（如 16、8），或训练时加 `--freeze`（只训练分类层，显存占用很小）。
2. **报错 `cudnn64_8.dll ... is not configured correctly`**：Windows 版飞桨缺少 cuDNN 运行库，执行一次
   ```bat
   py -3.12 fix_cudnn.py
   ```
   它会自动从国内镜像安装 cuDNN 并告知飞桨（之后 `--device auto` 就会用 GPU）。不修也能跑，系统会自动退回 CPU 训练，只是慢很多。
3. **摄像头不可用**：运行 `check_camera.py` 看提示；不影响系统功能，用 `--source demostream`（测试集模拟实时画面）或手机录制的 mp4 演示即可。
4. **提示「未找到人脸检测模型 haarcascade_frontalface_default.xml」**：本项目已在 `models/` 中自带该文件；若仍提示缺失，可运行 `get_face_model.py` 重新下载。
5. **第一次运行 `app.py` 启动较慢**：Gradio 首次导入需要编译较多模块（约 1~2 分钟），属正常现象，之后启动会快很多。
6. **重新划分数据集**：`py -3.12 prepare_dataset.py --force`；想用全部图片训练：`--max-per-class 0`（更慢、精度略高）。
7. **想用自己/同学的照片测试**：把照片放进 `demo_images\`，然后 `py -3.12 predict.py --image demo_images`。
8. **模型文件**：训练好的权重是 `outputs/best_model.pdparams`，所有推理脚本默认加载它；如需更换路径，可使用 `--model` 参数。

## 十二、说明

- 仓库中的 `models/*.xml` 为 OpenCV 官方 Haar 级联分类器文件（BSD-3-Clause 许可），用于人脸检测；
- 数据集与训练权重不随仓库分发，获取方式见上文；
- 本项目用于学习、毕业设计与技术演示，不建议直接用于医疗、安防等对准确性要求极高的生产场景。

## 十三、致谢

- [PaddlePaddle](https://www.paddlepaddle.org.cn/)
- [OpenCV](https://opencv.org/)
- [Gradio](https://www.gradio.app/)
- Face Mask Detection 公开数据集及其维护者

