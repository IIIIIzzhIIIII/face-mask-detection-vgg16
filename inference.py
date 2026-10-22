#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py —— 人脸检测 + 口罩识别 + 结果绘制（app.py 与 camera_demo.py 共用）

流程：先用 OpenCV 的 Haar 级联检测人脸框 -> 把人脸区域送入 VGG16 分类
      若没检测到人脸，则对整幅画面分类（保证任何情况下都有输出）
这样既符合"口罩佩戴检测"的语义，又能适配摄像头画面里有多个人脸的情况。
"""
import os
import glob

import cv2
import numpy as np
from PIL import Image

from predict import predict_image
from utils import BASE_DIR, CLASS_CN, draw_chinese_text

MAX_FACES = 6          # 单帧最多处理的人脸数
HAAR_NAME = "haarcascade_frontalface_default.xml"
# 优先使用项目自带的人脸检测模型文件（models/ 目录），再回退到 OpenCV 自带目录
HAAR_DIRS = [os.path.join(BASE_DIR, "models"), os.path.join(BASE_DIR, "demo_images")]
_cascade = None


def find_haar_path():
    """按优先级查找 Haar 人脸检测模型文件（不同 OpenCV 版本自带路径不一致，甚至没有）"""
    cands = [os.path.join(d, HAAR_NAME) for d in HAAR_DIRS]
    try:
        cands.append(os.path.join(cv2.data.haarcascades, HAAR_NAME))
    except Exception:
        pass
    try:
        cands.append(cv2.samples.findFile(HAAR_NAME))
    except Exception:
        pass
    try:                       # 在 OpenCV 安装目录里兜底搜索
        base = os.path.dirname(os.path.abspath(cv2.__file__))
        cands += glob.glob(os.path.join(base, "**", "haarcascade_frontalface*.xml"), recursive=True)
    except Exception:
        pass
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def get_face_cascade():
    """
    加载人脸检测器。找不到模型文件时返回 None，
    此时 analyze_frame 会自动退化为"对整幅画面分类"，系统依然可用（不会崩溃）。
    """
    global _cascade
    if _cascade is None:
        path = find_haar_path()
        ctor = getattr(cv2, "CascadeClassifier", None)
        if ctor is None:
            print("[提示] 当前 OpenCV 版本已不提供 Haar 人脸检测接口（OpenCV 5 起移除）。")
            print("      系统将直接对整幅画面做口罩识别，功能不受影响；")
            print("      如需人脸框标注，可执行： py -3.12 -m pip install opencv-python==4.10.0.84")
            _cascade = False
        elif not path:
            print("[提示] 未找到人脸检测模型 %s。" % HAAR_NAME)
            print("       系统将直接对整幅画面做口罩识别，功能不受影响。")
            print("       如需按人脸框标注，请运行： py -3.12 get_face_model.py")
            _cascade = False
        else:
            _cascade = ctor(path)
            if _cascade.empty():
                _cascade = False
    return _cascade if _cascade is not False else None


def detect_faces(bgr, scale_factor=1.1, min_neighbors=5):
    """返回人脸框列表 [(x1, y1, x2, y2), ...]，按面积从大到小排序；无检测器时返回空列表"""
    cascade = get_face_cascade()
    if cascade is None:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)                      # 直方图均衡，提升暗光下的检出率
    h, w = gray.shape[:2]
    min_side = max(36, int(min(h, w) * 0.13))
    faces = cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=(min_side, min_side))
    if len(faces) == 0:
        return []
    faces = sorted(faces, key=lambda f: int(f[2]) * int(f[3]), reverse=True)[:MAX_FACES]
    boxes = []
    for (x, y, fw, fh) in faces:
        # 稍微向外扩一点，保证口罩区域被完整包含
        mx, my = int(fw * 0.12), int(fh * 0.12)
        x1, y1 = max(0, int(x) - mx), max(0, int(y) - my)
        x2, y2 = min(w, int(x + fw) + mx), min(h, int(y + fh) + my)
        if x2 - x1 > 20 and y2 - y1 > 20:
            boxes.append((x1, y1, x2, y2))
    return boxes


def analyze_frame(model, bgr, threshold=0.6, use_face_detect=True):
    """
    对一帧画面做人脸检测 + 口罩识别
    返回 results 列表，每项：
        {"box": (x1,y1,x2,y2) 或 None, "label": 0/1, "conf": float,
         "probs": [p_with, p_without], "cost_ms": float}
    """
    results = []
    boxes = detect_faces(bgr) if use_face_detect else []
    if boxes:
        for (x1, y1, x2, y2) in boxes:
            crop = bgr[y1:y2, x1:x2]
            pil = Image.fromarray(crop[:, :, ::-1])          # BGR -> RGB
            idx, conf, prob, cost = predict_image(model, pil)
            results.append({"box": (x1, y1, x2, y2), "label": idx, "conf": conf,
                            "probs": prob.tolist(), "cost_ms": cost})
    else:
        pil = Image.fromarray(bgr[:, :, ::-1])
        idx, conf, prob, cost = predict_image(model, pil)
        results.append({"box": None, "label": idx, "conf": conf,
                        "probs": prob.tolist(), "cost_ms": cost})
    # 低置信度时按阈值判定为"不确定"，提示用户靠近/正对镜头
    for r in results:
        r["uncertain"] = r["conf"] < threshold
    return results


def draw_face_results(bgr, results, font_size=22):
    """把识别结果画到画面上（中文，"绿=已佩戴 / 红=未佩戴 / 橙=不确定"）"""
    texts = []
    for r in results:
        if r.get("uncertain"):
            color, name = (0, 165, 255), "无法确定"
        elif r["label"] == 0:
            color, name = (80, 200, 60), CLASS_CN[0]
        else:
            color, name = (60, 60, 235), CLASS_CN[1]
        label = "%s %.0f%%" % (name, r["conf"] * 100)
        if r["box"] is not None:
            x1, y1, x2, y2 = r["box"]
            thickness = max(2, int(font_size / 9))
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)
            texts.append((label, (x1 + 2, max(0, y1 - font_size - 8)), color, font_size))
        else:
            texts.append((label, (12, 12), color, font_size + 6))
    return draw_chinese_text(bgr, texts)


def summarize(results):
    """统计本帧结果：返回 (已佩戴人数, 未佩戴口罩人数)"""
    with_mask = sum(1 for r in results if (not r.get("uncertain")) and r["label"] == 0)
    without = sum(1 for r in results if (not r.get("uncertain")) and r["label"] == 1)
    return with_mask, without


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def imread_unicode(path):
    """读取图片（支持中文路径，cv2.imread 在 Windows 中文路径下会失败）"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    """保存图片（支持中文路径）"""
    try:
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return False
