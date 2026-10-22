#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py —— 口罩佩戴检测系统（Web 界面版，Gradio）

功能：
    上传本地图片，或直接调用摄像头/手机拍照 -> 自动检测画面中的人脸 ->
    VGG16 判断是否佩戴口罩 -> 返回标注后的图片和各类别置信度，并可保存记录。

用法：
    py -3.12 app.py                     # 默认 http://127.0.0.1:7860 并自动打开浏览器
    py -3.12 app.py --port 7861 --share
"""
import argparse
import os
import time

import gradio as gr

from inference import analyze_frame, bgr_to_pil, draw_face_results, pil_to_bgr, summarize
from predict import load_model
from utils import (CLASS_CN, CLASS_NAMES, DEMO_IMG_DIR, MODEL_PATH, SCREENSHOT_DIR,
                   TEST_DIR, draw_chinese_text, ensure_dirs, list_images)

_MODEL = {"model": None, "device": None}


def get_model(model_path):
    """第一次调用时加载模型，之后复用（避免每次识别都读盘）"""
    if _MODEL["model"] is None:
        _MODEL["model"], _MODEL["device"] = load_model(model_path, "auto")
        print("[系统] 模型加载完成，运行设备：%s" % _MODEL["device"].upper())
    return _MODEL["model"]


def recognize(pil_img, threshold, use_face, model_path):
    if pil_img is None:
        return None, {"提示": 1.0}
    t0 = time.perf_counter()
    bgr = pil_to_bgr(pil_img)
    results = analyze_frame(get_model(model_path), bgr, threshold=float(threshold),
                            use_face_detect=bool(use_face))
    with_mask, without = summarize(results)
    cost = (time.perf_counter() - t0) * 1000
    font = max(20, int(bgr.shape[0] / 22))
    out = draw_face_results(bgr, results, font_size=font)
    info = "人脸 %d 个：已佩戴 %d 人，未佩戴 %d 人    推理耗时 %.1f ms" % (
        len(results), with_mask, without, cost)
    out = draw_chinese_text(out, [(info, (12, out.shape[0] - font - 14), (40, 40, 40), font)])

    # 置信度（多个目标时取平均）
    prob_sum, n = [0.0, 0.0], 0
    for r in results:
        prob_sum = [prob_sum[0] + r["probs"][0], prob_sum[1] + r["probs"][1]]
        n += 1
    n = max(n, 1)
    label_dict = {CLASS_CN[0]: prob_sum[0] / n, CLASS_CN[1]: prob_sum[1] / n}

    ensure_dirs()
    out_path = os.path.join(SCREENSHOT_DIR, "app_%s.jpg" % time.strftime("%Y%m%d_%H%M%S"))
    bgr_to_pil(out).save(out_path)
    print("[系统] 识别完成：%s  结果：%s  图片已保存 %s" % (info, label_dict, out_path))
    return bgr_to_pil(out), label_dict


def build_ui(default_model=MODEL_PATH):
    # 优先使用测试集样图；如果仓库里没有下载数据集，则回退到 demo_images
    example_paths = []
    for cls in CLASS_NAMES:
        example_paths += list_images(os.path.join(TEST_DIR, cls))[:3]
    if not example_paths:
        example_paths = list_images(DEMO_IMG_DIR)[:6]
    examples = [[p, 0.6, True, default_model] for p in example_paths]

    with gr.Blocks(title="口罩佩戴检测系统") as demo:
        gr.Markdown(
            "## 基于 PaddlePaddle 的口罩佩戴检测系统\n"
            "模型：**VGG16**（13 层卷积 + 3 层全连接，ImageNet 预训练迁移学习）　"
            "流程：人脸检测 → 口罩识别 → 结果标注\n\n"
            "**使用说明**：左侧上传图片，或点击图片框里的相机图标拍照（手机访问同一局域网地址即可用手机拍），"
            "然后点「开始识别」。")
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="输入图片 / 拍照")
                thr = gr.Slider(minimum=0.5, maximum=0.99, value=0.6, step=0.01,
                                label="置信度阈值（低于该值时提示：无法确定）")
                face_chk = gr.Checkbox(value=True, label="先检测人脸（关闭则对整张图分类）")
                model_path = gr.Textbox(value=default_model, label="模型文件路径", visible=False)
                with gr.Row():
                    btn = gr.Button("开始识别")
                    clear = gr.Button("清空")
            with gr.Column(scale=1):
                img_out = gr.Image(type="pil", label="识别结果")
                lab_out = gr.Label(num_top_classes=2, label="各类别置信度")
        btn.click(recognize, [img_in, thr, face_chk, model_path], [img_out, lab_out])
        clear.click(lambda: (None, None, None), None, [img_in, img_out, lab_out])
        if examples:
            gr.Examples(examples=examples, inputs=[img_in, thr, face_chk, model_path],
                        outputs=[img_out, lab_out], cache_examples=False,
                        label="测试集示例（点击即可快速验证）")
        gr.Markdown("识别记录自动保存到 `outputs/screenshots/` 目录，可用于撰写论文的实验结果章节。")
    return demo


def main():
    ap = argparse.ArgumentParser(description="口罩佩戴检测系统 Web 界面")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成临时公网链接（手机也可访问）")
    args = ap.parse_args()
    ensure_dirs()

    if not os.path.exists(args.model):
        print("[错误] 找不到模型 %s，请先运行 train.py 训练" % args.model)
        return
    demo = build_ui(args.model)
    print("=" * 62)
    print("正在启动 Web 界面，请用浏览器打开： http://%s:%d" % (args.host, args.port))
    print("按 Ctrl+C 可关闭服务")
    print("=" * 62)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
