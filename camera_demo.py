#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
camera_demo.py —— 实时口罩佩戴检测演示（答辩现场用）

支持多种输入源（识别流程完全相同，只换输入源，无需改代码）：
    --source 0            笔记本/外接摄像头（设备索引）
    --source video.mp4    手机录制的视频文件
    --source folder       一个图片文件夹（逐张当视频帧播放）
    --source demostream   用测试集图片合成"实时视频"（本机默认，内置摄像头硬件不可用时的替代方案）
    --source screen       识别屏幕画面（演示时把视频/照片全屏播放即可）

常用参数：
    --no-face             关闭人脸检测（直接对整帧分类）
    --threshold 0.6       置信度阈值，低于该值显示"无法确定"
    --max-frames 300      处理多少帧后结束（0=一直运行）
    --no-window           不弹窗口，只保存截图和录像
    --out-video auto      保存演示录像到 outputs/videos/
    --fps 15              demostream / screen 的模拟帧率

运行时快捷键： Q/ESC 退出    S 保存截图    R 开始/停止录像
"""
import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
import time
from collections import deque

import cv2
import numpy as np

from inference import (analyze_frame, draw_face_results, imread_unicode, imwrite_unicode,
                       summarize)
from predict import load_model
from utils import (CLASS_NAMES, OUT_DIR, SCREENSHOT_DIR, TEST_DIR, VIDEO_DIR, MODEL_PATH,
                   draw_chinese_text, ensure_dirs, list_images)

WINDOW_NAME = "Mask Detection System (VGG16)  -  press Q to quit"


def fit_side(img, max_side=960):
    """按比例缩放，使最长边不超过 max_side（提高实时性）"""
    h, w = img.shape[:2]
    scale = float(max_side) / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def shift_image(img, dx):
    """水平平移少量像素，模拟摄像头轻微晃动（让合成视频更像真实画面）"""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def smooth_labels(results, histories, window):
    """
    时序平滑：同一个人脸在连续若干帧里取"出现次数最多的类别"，
    避免画面抖动导致识别结果频繁跳变（真实系统里必须有这一步）。
    """
    if window <= 1:
        return results
    for i, r in enumerate(results):
        d = histories.setdefault(i, deque(maxlen=window))
        d.append(r["label"])
        values = list(d)
        r["label"] = max(set(values), key=values.count)
    return results


def draw_hud(frame, hud_lines, line_h=30):
    """在画面底部绘制半透明信息栏（中文）"""
    h, w = frame.shape[:2]
    bar_h = min(line_h * len(hud_lines) + 12, h)
    region = frame[h - bar_h:h, :]
    overlay = region.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    frame[h - bar_h:h, :] = cv2.addWeighted(overlay, 0.6, region, 0.4, 0)
    texts = [(item[0], (12, h - bar_h + 6 + i * line_h), item[1], max(16, line_h - 8))
             for i, item in enumerate(hud_lines)]
    return draw_chinese_text(frame, texts)


def ascii_work_dir():
    """找一个可写且路径为纯 ASCII 的目录（本机用户名含中文，cv2.VideoWriter 存不了中文路径）"""
    for cand in [os.environ.get("PUBLIC"), "C:\\Users\\Public",
                 os.environ.get("ProgramData"), "C:\\Windows\\Temp"]:
        if cand and all(ord(c) < 128 for c in cand) and os.path.isdir(cand):
            try:
                test = os.path.join(cand, "_mask_write_test.tmp")
                with open(test, "w") as f:
                    f.write("ok")
                os.remove(test)
                return cand
            except Exception:
                continue
    return None


def safe_video_writer(path, fps, size):
    """
    创建视频写入器。cv2.VideoWriter 不支持中文路径，
    因此遇到中文路径时先写到纯英文目录，结束后再移动到目标路径。
    返回 (writer 或 None, 实际写入路径 或 None, 最终目标路径)
    """
    def try_open(p):
        fourcc = 0
        if hasattr(cv2, "VideoWriter_fourcc"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif hasattr(cv2.VideoWriter, "fourcc"):
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(p, fourcc, max(fps, 8), size)
        return writer if writer.isOpened() else None

    if not path:
        return None, None, None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if all(ord(c) < 128 for c in path):                 # 纯英文路径，直接写
        w = try_open(path)
        return (w, path, path) if w else (None, None, path)
    work = ascii_work_dir()                             # 中文路径：先写到英文目录，最后搬过去
    if not work:
        return None, None, path
    tmp = os.path.join(work, "mask_demo_tmp.mp4")
    w = try_open(tmp)
    if w is None:
        tmp = os.path.join(work, "mask_demo_tmp.avi")
        w = try_open(tmp)
    return w, (tmp if w else None), path


# ------------------------------------------------------------------ 输入源
class CaptureSource:
    """摄像头 / 视频文件（OpenCV 采集）"""
    def __init__(self, cap, name):
        self.cap = cap
        self.name = name

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.cap.release()


class ImageStreamSource:
    """
    图片文件夹 或 测试集 合成的"实时视频流"：
    逐张读图 -> 缩放 -> 轻微平移 -> 按指定帧率输出，模拟摄像头实时画面。
    """
    def __init__(self, images, name, fps=15.0, max_side=960, shuffle=True, seed=0):
        self.images = list(images)
        if shuffle:
            random.Random(seed).shuffle(self.images)
        self.name = name
        self.fps = max(float(fps), 1.0)
        self.max_side = max_side
        self.idx = 0
        self.offset = 0
        self.direction = 1

    def read(self):
        if not self.images:
            return False, None
        path = self.images[self.idx % len(self.images)]
        self.idx += 1
        img = imread_unicode(path)
        if img is None:
            return False, None
        img = fit_side(img, self.max_side)
        self.offset += self.direction * 3
        if abs(self.offset) > 21:
            self.direction *= -1
        img = shift_image(img, self.offset)
        time.sleep(1.0 / self.fps)
        return True, img

    def release(self):
        pass


class ScreenSource:
    """识别屏幕画面（Windows 用 PIL.ImageGrab 截图，无需额外依赖）"""
    def __init__(self, fps=12.0, max_side=960):
        self.fps = max(float(fps), 1.0)
        self.max_side = max_side
        self.name = "屏幕画面"

    def read(self):
        try:
            from PIL import ImageGrab
        except Exception as e:
            print("[错误] 无法捕获屏幕：%s" % e)
            return False, None
        arr = np.array(ImageGrab.grab())          # RGB
        bgr = arr[:, :, ::-1].copy()
        time.sleep(1.0 / self.fps)
        return True, fit_side(bgr, self.max_side)

    def release(self):
        pass


def build_source(args):
    """根据 --source 参数创建输入源对象"""
    src = str(args.source).strip()
    if src.isdigit():                                       # 摄像头索引
        backend = getattr(cv2, "CAP_DSHOW", 0) if os.name == "nt" else 0
        cap = cv2.VideoCapture(int(src), backend)
        if not cap.isOpened():
            print("[错误] 打不开摄像头 %s。请先运行 check_camera.py 自检，" % src)
            print("       或改用：py -3.12 camera_demo.py --source demostream")
            sys.exit(2)
        return CaptureSource(cap, "摄像头 %s" % src)

    low = src.lower()
    if low in ("demostream", "demo", "stream"):              # 测试集合成的实时视频
        images = []
        for cls in CLASS_NAMES:
            images += list_images(os.path.join(TEST_DIR, cls))
        if not images:
            print("[错误] 测试集里没有图片，请先运行 prepare_dataset.py")
            sys.exit(2)
        print("[输入源] demostream：用 %d 张测试集图片合成实时视频" % len(images))
        return ImageStreamSource(images, "demostream(测试集模拟实时画面)",
                                 fps=args.fps, max_side=args.max_side)

    if low == "screen":
        return ScreenSource(fps=args.fps, max_side=args.max_side)

    if os.path.isdir(src):                                   # 图片文件夹
        images = list_images(src)
        if not images:
            print("[错误] 文件夹里没有图片：%s" % src)
            sys.exit(2)
        return ImageStreamSource(images, "图片文件夹 %s" % src, fps=args.fps,
                                 max_side=args.max_side, shuffle=False)

    if os.path.isfile(src):                                  # 视频文件
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print("[错误] 打不开视频文件：%s" % src)
            sys.exit(2)
        return CaptureSource(cap, "视频文件 %s" % os.path.basename(src))

    print("[错误] 无法识别的输入源：%s" % src)
    print("       可用：0/1(摄像头) | demostream | screen | 视频文件 | 图片文件夹")
    sys.exit(2)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="实时口罩佩戴检测演示（多输入源）")
    ap.add_argument("--source", default="demostream",
                    help="输入源：0/1(摄像头索引) | 视频文件 | 图片文件夹 | demostream | screen")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="置信度阈值，低于该值显示 无法确定")
    ap.add_argument("--no-face", action="store_true", help="关闭人脸检测，直接对整帧分类")
    ap.add_argument("--smooth", type=int, default=5, help="时序平滑帧数（防结果跳变）")
    ap.add_argument("--fps", type=float, default=15.0, help="demostream/screen 的帧率")
    ap.add_argument("--max-side", type=int, default=960, help="画面最长边，越小越快")
    ap.add_argument("--max-frames", type=int, default=0, help="处理多少帧后结束，0=不限")
    ap.add_argument("--no-window", action="store_true", help="不弹窗口，只保存文件")
    ap.add_argument("--out-video", default="auto", help="auto=自动保存演示录像 / 具体路径 / none")
    ap.add_argument("--auto-save-interval", type=float, default=3.0, help="自动截图间隔（秒）")
    ap.add_argument("--max-shots", type=int, default=15, help="最多自动截图张数")
    args = ap.parse_args()
    ensure_dirs()

    model, device = load_model(args.model, args.device)
    source = build_source(args)
    print("=" * 68)
    print("模型文件：%s" % args.model)
    print("运行设备：%s    输入源：%s" % (device.upper(), source.name))
    print("人脸检测：%s    置信度阈值：%.2f    时序平滑：%d 帧"
          % ("开启" if not args.no_face else "关闭", args.threshold, args.smooth))
    if not args.no_window:
        print("快捷键：Q/ESC 退出    S 保存截图    R 开始/停止录像")
    print("=" * 68)

    # ---------- 录像目标路径 ----------
    src = str(args.source).strip()
    out_video = None
    if args.out_video.lower() == "auto":
        if src.isdigit():
            out_video = os.path.join(VIDEO_DIR, "demo_camera_%s.mp4" % src)
        elif src.lower() in ("demostream", "demo", "stream"):
            out_video = os.path.join(VIDEO_DIR, "demo_demostream.mp4")
        elif src.lower() == "screen":
            out_video = os.path.join(VIDEO_DIR, "demo_screen.mp4")
        else:
            out_video = os.path.join(VIDEO_DIR, "demo_video.mp4")
    elif args.out_video.lower() not in ("none", "no", "off", "false"):
        out_video = args.out_video

    writer, tmp_video, final_video = None, None, None
    recording = bool(out_video)
    histories = {}
    stats = {"with": 0, "without": 0, "uncertain": 0}
    shots, last_shot = 0, 0.0
    fps_ema, infer_ema, infer_count = 0.0, 0.0, 0
    frame_i, fail_streak = 0, 0
    t_start = time.time()
    t_prev = t_start

    log_path = os.path.join(OUT_DIR, "realtime_log.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8-sig")
    log_w = csv.writer(log_f)
    log_w.writerow(["时间", "输入源", "识别结果", "置信度", "已戴口罩概率", "未戴口罩概率", "单帧推理ms"])
    last_log = 0.0

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                fail_streak += 1
                limit = 60 if isinstance(source, CaptureSource) else 30
                if fail_streak > limit:
                    print("[结束] 输入源已读取完毕")
                    break
                time.sleep(0.02)
                continue
            fail_streak = 0
            frame_i += 1

            # ---- 核心：人脸检测 + 口罩识别 ----
            t0 = time.perf_counter()
            results = analyze_frame(model, frame, threshold=args.threshold,
                                    use_face_detect=not args.no_face)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_ema = infer_ms if infer_count == 0 else infer_ema * 0.8 + infer_ms * 0.2
            infer_count += 1
            smooth_labels(results, histories, args.smooth)         # 时序平滑
            w_cnt, wo_cnt = summarize(results)
            stats["with"] += w_cnt
            stats["without"] += wo_cnt
            stats["uncertain"] += sum(1 for r in results if r.get("uncertain"))

            # ---- 绘制结果 ----
            frame = draw_face_results(frame, results, font_size=max(18, int(frame.shape[1] / 48)))
            if wo_cnt > 0:
                frame = draw_chinese_text(frame, [("警告：检测到未佩戴口罩人员！", (12, 12),
                                                  (60, 60, 235), max(20, int(frame.shape[1] / 34)))])
            now = time.time()
            inst_fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            fps_ema = inst_fps if fps_ema == 0 else fps_ema * 0.9 + inst_fps * 0.1
            frame = draw_hud(frame, [
                ("输入源：%s    第 %d 帧" % (source.name, frame_i), (235, 235, 235)),
                ("本帧：已佩戴 %d 人 / 未佩戴 %d 人    累计 %d / %d 人次"
                 % (w_cnt, wo_cnt, stats["with"], stats["without"]), (120, 255, 120)),
                ("FPS %.1f    推理 %.0f ms    设备 %s%s"
                 % (fps_ema, infer_ema, device.upper(),
                    "    ● 录像中" if (recording and writer is not None) else ""), (120, 230, 255)),
            ], line_h=max(22, int(frame.shape[0] / 26)))

            # ---- 显示与快捷键 ----
            if not args.no_window:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("[退出] 用户按键退出")
                    break
                if key in (ord("s"), ord("S")):
                    p = os.path.join(SCREENSHOT_DIR, "shot_%s_%03d.jpg"
                                     % (time.strftime("%Y%m%d_%H%M%S"), frame_i))
                    if imwrite_unicode(p, frame):
                        shots += 1
                        print("[截图] %s" % p)
                if key in (ord("r"), ord("R")):
                    recording = not recording
                    print("[录像] %s" % ("继续录制" if recording else "已停止录制"))

            # ---- 录制演示视频 ----
            if out_video and recording:
                if writer is None:
                    writer, tmp_video, final_video = safe_video_writer(
                        out_video, args.fps, (frame.shape[1], frame.shape[0]))
                    if writer is None:
                        print("[警告] 无法创建视频文件，已关闭录像功能。目标路径：%s" % out_video)
                        out_video, recording = None, False
                if writer is not None:
                    writer.write(frame)

            # ---- 自动截图（有把握的结果每隔几秒存一张，方便写论文/答辩） ----
            if shots < args.max_shots and results and (now - last_shot) >= args.auto_save_interval:
                best = max(results, key=lambda r: r["conf"])
                if best["conf"] >= args.threshold:
                    p = os.path.join(SCREENSHOT_DIR, "realtime_%s_%03d.jpg"
                                     % (time.strftime("%Y%m%d_%H%M%S"), shots))
                    if imwrite_unicode(p, frame):
                        shots += 1
                        last_shot = now
                        print("[自动截图 %d/%d] %s" % (shots, args.max_shots, p))

            # ---- 写入识别日志（每秒最多一条） ----
            if results and (now - last_log) >= 1.0:
                last_log = now
                for r in results:
                    log_w.writerow([time.strftime("%H:%M:%S"), source.name, CLASS_CN[r["label"]],
                                    "%.4f" % r["conf"], "%.4f" % r["probs"][0],
                                    "%.4f" % r["probs"][1], "%.1f" % r["cost_ms"]])

            if args.max_frames and frame_i >= args.max_frames:
                print("[结束] 已达到设定帧数 %d" % args.max_frames)
                break
    except KeyboardInterrupt:
        print("\n[退出] 收到 Ctrl+C")
    finally:
        try:
            log_f.close()
        except Exception:
            pass
        try:
            source.release()
        except Exception:
            pass
        if not args.no_window:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
            if tmp_video and final_video and os.path.abspath(tmp_video) != os.path.abspath(final_video):
                try:
                    shutil.move(tmp_video, final_video)
                except Exception as e:
                    print("[警告] 移动录像文件失败：%s（临时文件在 %s）" % (e, tmp_video))
            print("[录像] 演示录像已保存：%s" % (final_video or tmp_video))

        dur = max(time.time() - t_start, 1e-6)
        print("=" * 68)
        print("演示结束：共处理 %d 帧，耗时 %.1f 秒，平均 %.1f FPS" % (frame_i, dur, frame_i / dur))
        print("累计识别：已佩戴口罩 %d 人次，未佩戴口罩 %d 人次，无法确定 %d 人次"
              % (stats["with"], stats["without"], stats["uncertain"]))
        print("截图目录：%s（共 %d 张）" % (SCREENSHOT_DIR, shots))
        print("识别日志：%s" % log_path)


if __name__ == "__main__":
    main()
