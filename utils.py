#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py —— 公共工具模块
包含：路径常量、类别定义、中文字体处理、matplotlib 中文设置、图片网格拼接

本文件被 train.py / evaluate.py / predict.py / app.py / camera_demo.py 共用，
保证"训练 → 评估 → 推理"三阶段的预处理完全一致（这是模型效果的关键）。
"""
import os
import glob

# ---------------------------------------------------------------- 路径常量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")            # 原始数据 + 解压目录
DATA_ROOT = os.path.join(DATASET_DIR, "facemask")          # 划分好的数据集根目录
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")
OUT_DIR = os.path.join(BASE_DIR, "outputs")                # 所有图表/指标/日志输出
SCREENSHOT_DIR = os.path.join(OUT_DIR, "screenshots")      # 实时识别截图
VIDEO_DIR = os.path.join(OUT_DIR, "videos")                # 演示录像
MODEL_PATH = os.path.join(OUT_DIR, "best_model.pdparams")  # 最佳模型权重
DEMO_IMG_DIR = os.path.join(BASE_DIR, "demo_images")       # 自己拍的照片放这里

# 类别定义（与目录名一一对应，顺序不可改：训练/评估/推理三处必须一致）
CLASS_NAMES = ["with_mask", "without_mask"]      # 文件夹名
CLASS_CN = ["已佩戴口罩", "未佩戴口罩"]            # 中文显示名
CLASS_CN_SHORT = ["已佩戴", "未佩戴"]
IMG_SIZE = 224                                   # VGG16 标准输入尺寸
IMAGENET_MEAN = [0.485, 0.456, 0.406]            # 归一化均值（ImageNet 标准）
IMAGENET_STD = [0.229, 0.224, 0.225]             # 归一化方差

def _prepare_cudnn():
    """
    Windows 版飞桨需要额外的 cuDNN 运行库，否则 GPU 不可用
    （报错：The third-party dynamic library (cudnn64_8.dll) ... is not configured correctly）。
    如果 site-packages 里通过 pip 装了 nvidia-cudnn-cu11，就把它的 bin 目录加入 DLL 搜索路径，
    这样所有脚本都不用再手动配置环境变量。
    """
    if os.name != "nt":
        return
    try:
        import site as _site
        dirs = []
        for sp in list(_site.getsitepackages()) + [os.path.dirname(os.__file__)]:
            dirs += glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
        for d in dirs:
            if os.path.isdir(d) and glob.glob(os.path.join(d, "cudnn*.dll")):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
    except Exception:
        pass


_prepare_cudnn()


def cuda_usable():
    """
    安全判断当前环境能否真正使用 GPU（不会真的跑 GPU 算子）。
    为什么要这么判断：
        只判断"编译了 CUDA / 能查到设备数"不够——Windows 版飞桨还需要 cuDNN(cudnn64_8.dll)
        和 cuBLAS(cublasLt64_11.dll)，缺任何一个都不能用 GPU；
        而缺库时飞桨的底层加载器会直接结束进程（不是抛 Python 异常），
        所以这里改用"检查 DLL 文件是否存在"，既能正确判断又不会把程序弄崩。
    """
    try:
        import paddle
        if not paddle.device.is_compiled_with_cuda():
            return False
        if int(paddle.device.cuda.device_count()) <= 0:
            return False
    except Exception:
        return False
    if os.name != "nt":
        return True
    # 飞桨实际会搜索的位置：它自己的 libs 目录、CUDA 安装目录
    # （放在 site-packages\nvidia\... 里它搜不到，所以必须链接/复制到 libs 才行）
    libs = os.path.join(os.path.dirname(os.path.abspath(paddle.__file__)), "libs")
    search = [libs]
    search += glob.glob(os.path.join(
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        "NVIDIA GPU Computing Toolkit", "CUDA", "*", "bin"))
    for name in ("cudnn64_8.dll", "cublasLt64_11.dll"):
        if not any(os.path.exists(os.path.join(d, name)) for d in search):
            return False
    return True


def pick_device(device="auto"):
    """统一处理 --device 参数：auto 时自动选择可用的设备"""
    if device == "auto":
        return "gpu" if cuda_usable() else "cpu"
    if device == "gpu" and not cuda_usable():
        print("[警告] 指定使用 GPU，但当前 GPU 不可用（多数情况是缺少 cuDNN），已自动改用 CPU。")
        print("       修复方法：运行 py -3.12 fix_cudnn.py")
        return "cpu"
    return device


# ------------------------------------------------------------ 目录自动创建
def ensure_dirs():
    """保证所有输出目录存在（反复调用安全）"""
    for d in [DATASET_DIR, OUT_DIR, SCREENSHOT_DIR, VIDEO_DIR, DEMO_IMG_DIR]:
        os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------ 中文字体处理
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑（本机已确认存在）
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体（本机已确认存在）
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",   # Linux 备用
]


def find_chinese_font():
    """返回第一个存在的中文字体文件路径，找不到返回 None"""
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def get_pil_font(size=24):
    """获取指定字号的中文 PIL 字体对象（找不到时退回默认字体，不会报错）"""
    from PIL import ImageFont
    path = find_chinese_font()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_chinese_text(img_bgr, texts):
    """
    在 OpenCV 的 BGR 图像上绘制中文文字（OpenCV 自带 putText 不支持中文，故用 PIL 绘制）
    texts: [(文本, (x, y), (B, G, R)), ...]  坐标为文字左上角
    返回：绘制后的 BGR 图像
    """
    import numpy as np
    from PIL import Image, ImageDraw
    if not texts:
        return img_bgr
    pil_img = Image.fromarray(img_bgr[:, :, ::-1])       # BGR -> RGB
    draw = ImageDraw.Draw(pil_img)
    for item in texts:
        text, pos, color = item[0], item[1], item[2]
        size = item[3] if len(item) > 3 else 22
        draw.text(pos, text, font=get_pil_font(size), fill=(int(color[2]), int(color[1]), int(color[0])))
    return np.array(pil_img)[:, :, ::-1].copy()          # RGB -> BGR


def setup_matplotlib_chinese():
    """让 matplotlib 能正常显示中文（找不到中文字体时自动退回英文标签）"""
    import matplotlib
    from matplotlib import font_manager
    path = find_chinese_font()
    if path:
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            matplotlib.rcParams["font.sans-serif"] = [name, "SimHei", "Microsoft YaHei"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            pass
    return False


# ------------------------------------------------------------ 图片网格拼接
def make_grid_image(paths, labels, cols=4, cell=224, title=None):
    """
    把多张图片拼成一张网格图（用于自查"数据集到底长什么样"）
    paths : 图片路径列表
    labels: 与 paths 对应的文字（如类别中文名）
    返回：PIL Image
    """
    from PIL import Image, ImageDraw
    rows = (len(paths) + cols - 1) // cols
    header = 34
    canvas = Image.new("RGB", (cols * cell, rows * (cell + header)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = get_pil_font(20)
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        try:
            im = Image.open(p).convert("RGB").resize((cell, cell))
        except Exception:
            im = Image.new("RGB", (cell, cell), (200, 200, 200))
        canvas.paste(im, (c * cell, r * (cell + header) + header))
        draw.text((c * cell + 6, r * (cell + header) + 6), str(labels[i]), font=font, fill=(0, 0, 0))
    return canvas


def list_images(folder, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """列出文件夹下所有图片（不递归）"""
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(folder, "*" + e)))
        out.extend(glob.glob(os.path.join(folder, "*" + e.upper())))
    return sorted(set(out))


def count_samples(root):
    """统计 root/{train,val,test}/{with_mask,without_mask} 下的图片数量"""
    stat = {}
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            d = os.path.join(root, split, cls)
            n = len(list_images(d)) if os.path.isdir(d) else 0
            stat[f"{split}/{cls}"] = n
    return stat
