"""Shared colorization service for CLI and Gradio UI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Prefer D: for caches / temps
_TEMP = Path(r"D:\AI_Temp\colorize_ui")
_TEMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(Path(r"D:\AI_Temp") / "torch_cache"))
os.environ.setdefault("FASTAI_HOME", str(Path(r"D:\AI_Temp") / "fastai_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["DEOLDIFY_MODEL_DIR"] = str(MODELS)

if not hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals = lambda globals_list: None  # type: ignore

_DEVICE = None
_original_torch_load = torch.load


def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _patched_torch_load(*args, **kwargs):
    kwargs.pop("weights_only", None)
    if get_device().type == "cuda":
        kwargs.setdefault("map_location", get_device())
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def imread_unicode(filepath: str):
    with open(filepath, "rb") as f:
        data = f.read()
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取: {filepath}")
    return img


def imwrite_unicode(filepath: str, img) -> None:
    ext = os.path.splitext(filepath)[1].lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise IOError(f"编码失败: {filepath}")
    with open(filepath, "wb") as f:
        f.write(buf.tobytes())


def mean_chroma(img_bgr: np.ndarray) -> float:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.sqrt((lab[:, :, 1] - 128) ** 2 + (lab[:, :, 2] - 128) ** 2).mean())


def neutralize_yellow(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return cv2.cvtColor(lab[:, :, 0], cv2.COLOR_GRAY2BGR)


def auto_contrast_luma(img_bgr: np.ndarray, clip_percent: float = 0.8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    lo = np.percentile(L, clip_percent)
    hi = np.percentile(L, 100.0 - clip_percent)
    if hi <= lo + 1:
        return img_bgr
    lab[:, :, 0] = np.clip((L - lo) * (255.0 / (hi - lo)), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def prepare_input(img_bgr: np.ndarray, deyellow: bool, autocontrast: bool) -> np.ndarray:
    out = img_bgr
    if deyellow:
        out = neutralize_yellow(out)
    if autocontrast:
        out = auto_contrast_luma(out)
    return out


def crop_photo_region(img_bgr: np.ndarray, enabled: bool = True):
    """仅在确有白页边时裁切；无页边则原样返回，避免贴回灰边框。"""
    h, w = img_bgr.shape[:2]
    if not enabled:
        return img_bgr, (0, h, 0, w)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 近白像素视为页边
    white = gray >= 245
    white_ratio = float(white.mean())
    # 几乎没有白边（普通照片）→ 不裁
    if white_ratio < 0.02:
        return img_bgr, (0, h, 0, w)

    mask = (~white).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 1000:
        return img_bgr, (0, h, 0, w)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    # 轻微外扩，避免切到画面；不再向内缩（向内缩会在贴回时留下灰框）
    pad_y = max(2, (y1 - y0) // 200)
    pad_x = max(2, (x1 - x0) // 200)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)

    # 裁完几乎还是整图 → 不裁
    if (y1 - y0) > h * 0.97 and (x1 - x0) > w * 0.97:
        return img_bgr, (0, h, 0, w)
    if (y1 - y0) < h * 0.45 or (x1 - x0) < w * 0.45:
        return img_bgr, (0, h, 0, w)
    return img_bgr[y0:y1, x0:x1], (y0, y1, x0, x1)


def paste_photo_region(full_bgr: np.ndarray, crop_bgr: np.ndarray, box) -> np.ndarray:
    """贴回时：页边用浅色纸底，不再用整图灰度，避免出现黑白相框。"""
    y0, y1, x0, x1 = box
    # 浅灰白页边（接近相纸），不是原图去色
    out = np.full_like(full_bgr, 245)
    ch, cw = crop_bgr.shape[:2]
    if (ch, cw) != (y1 - y0, x1 - x0):
        crop_bgr = cv2.resize(crop_bgr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    out[y0:y1, x0:x1] = crop_bgr
    return out


def _sky_soft_mask(L: np.ndarray) -> np.ndarray:
    """偏亮、偏上、纹理少 → 天空候选（用于限 chroma / 压色块）。"""
    h, w = L.shape
    top = np.linspace(1.0, 0.18, h, dtype=np.float32)[:, None]
    top = np.broadcast_to(top, (h, w)).copy()
    bright = np.clip((L - 145.0) / 70.0, 0.0, 1.0)
    Lf = cv2.GaussianBlur(L.astype(np.float32), (0, 0), 2.5)
    var = cv2.blur((L.astype(np.float32) - Lf) ** 2, (21, 21))
    smooth = 1.0 - np.clip(var / 90.0, 0.0, 1.0)
    m = bright * top * (0.35 + 0.65 * smooth)
    return cv2.GaussianBlur(m, (0, 0), max(2.0, min(h, w) / 180.0))


def _fix_sky_yellow_green(L: np.ndarray, a: np.ndarray, b: np.ndarray):
    """压制天空黄/黄绿大色块，尽量不动水面与船体。"""
    h, w = L.shape
    sky_base = _sky_soft_mask(L)
    # 黄 / 黄绿偏离（阈值放宽，抓大色块）
    yellow = np.clip((b - 128.0) / 16.0, 0.0, 1.0)
    greenish = np.clip((128.0 - a) / 14.0, 0.0, 1.0) * np.clip((b - 120.0) / 14.0, 0.0, 1.0)
    cast = np.clip(yellow * 0.9 + greenish * 1.2, 0.0, 1.0)
    # 大面积平滑色块：膨胀连通后再模糊
    blotch = (sky_base * cast > 0.22).astype(np.uint8) * 255
    k = max(5, (min(h, w) // 80) | 1)
    blotch = cv2.morphologyEx(blotch, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    blotch = cv2.dilate(blotch, np.ones((k, k), np.uint8), iterations=1)
    blotch = cv2.GaussianBlur(blotch.astype(np.float32) / 255.0, (0, 0), max(3.0, k / 2.0))
    sky = np.clip(sky_base * cast * 0.55 + blotch * sky_base * 0.85, 0.0, 1.0)
    # 几乎拉回中性偏冷蓝
    a = a * (1.0 - sky * 0.95) + 128.0 * (sky * 0.95)
    b = b * (1.0 - sky * 0.96) + 118.0 * (sky * 0.96)
    hi = np.clip((L - 155.0) / 75.0, 0.0, 1.0) * sky_base
    b = b - hi * np.clip(b - 128.0, 0.0, None) * 0.85
    a = a + hi * np.clip(120.0 - a, 0.0, None) * 0.45
    return a, b


def adjust_color(img_bgr: np.ndarray, chroma: float = 1.35, cool: float = 0.22) -> np.ndarray:
    chroma = float(np.clip(chroma, 0.5, 2.2))
    cool = float(np.clip(cool, 0.0, 1.0))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    cur = float(np.sqrt((a - 128) ** 2 + (b - 128) ** 2).mean())
    auto_boost = 1.55 if cur < 5 else (1.25 if cur < 9 else 1.0)
    chroma_eff = chroma * auto_boost
    # 天空区域少加浓度，避免错色被放大成大色块
    sky_m = _sky_soft_mask(L)
    gain = 1.0 + (chroma_eff - 1.0) * (1.0 - sky_m * 0.9)
    a = 128.0 + (a - 128.0) * gain
    b = 128.0 + (b - 128.0) * gain
    mean_b = float(b.mean())
    if mean_b > 130:
        b -= (mean_b - 128.0) * 0.7
    a, b = _fix_sky_yellow_green(L, a, b)
    if cool > 0:
        mean_a = float(a.mean())
        if mean_a > 132:
            a -= (mean_a - 128.0) * (0.45 * cool)
        a -= np.clip(a - 145.0, 0.0, None) * (0.5 * cool)
    L8 = np.clip(L, 0, 255).astype(np.uint8)
    edges = cv2.Canny(L8, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (5, 5), 0)
    a_s = cv2.bilateralFilter(a.astype(np.float32), 7, 18, 7)
    b_s = cv2.bilateralFilter(b.astype(np.float32), 7, 18, 7)
    a = a * (1.0 - edges * 0.7) + a_s * (edges * 0.7)
    b = b * (1.0 - edges * 0.7) + b_s * (edges * 0.7)
    return cv2.cvtColor(
        cv2.merge(
            [
                np.clip(L, 0, 255).astype(np.uint8),
                np.clip(a, 0, 255).astype(np.uint8),
                np.clip(b, 0, 255).astype(np.uint8),
            ]
        ),
        cv2.COLOR_LAB2BGR,
    )


class ColorizeEngine:
    """Lazy-load models and run image/video colorization."""

    def __init__(self):
        self._dd = {}
        self._deo = None
        self._deo_video = None

    def _resolve_dd_weight(self, kind: str) -> Path:
        if kind == "modelscope":
            cands = [
                MODELS / "ddcolor_modelscope.bin",
                MODELS / "ddcolor_modelscope.pt",
                ROOT / "modelscope" / "damo" / "cv_ddcolor_image-colorization" / "pytorch_model.pt",
            ]
        else:
            cands = [MODELS / "ddcolor_artistic.bin", MODELS / "ddcolor_artistic.pt"]
        for p in cands:
            if p.exists() and p.stat().st_size > 100_000_000:
                return p
        raise FileNotFoundError(f"缺少 DDColor 权重 ({kind})，请先运行 python download_models.py")

    def get_ddcolor(self, kind: str, input_size: int):
        key = (kind, int(input_size))
        if key not in self._dd:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            for mod in list(sys.modules):
                if mod == "ddcolor" or mod.startswith("ddcolor."):
                    del sys.modules[mod]
            from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

            weight = self._resolve_dd_weight(kind)
            model = build_ddcolor_model(
                DDColor,
                model_path=str(weight),
                input_size=int(input_size),
                model_size="large",
                device=get_device(),
            )
            self._dd[key] = ColorizationPipeline(model, input_size=int(input_size), device=get_device())
        return self._dd[key]

    def get_deoldify(self, artistic: bool = False, render_factor: int = 30):
        from deoldify.visualize import get_image_colorizer

        # Reload when artistic switches
        need_reload = self._deo is None or getattr(self, "_deo_artistic", None) != artistic
        if need_reload:
            self._deo = get_image_colorizer(root_folder=ROOT, artistic=artistic, render_factor=render_factor)
            self._deo_artistic = artistic
        self._deo.render_factor = render_factor
        return self._deo

    def get_deoldify_video(self, render_factor: int = 21):
        from deoldify.visualize import get_stable_video_colorizer

        if self._deo_video is None:
            self._deo_video = get_stable_video_colorizer(root_folder=ROOT, render_factor=render_factor)
        self._deo_video.vis.render_factor = render_factor
        return self._deo_video

    def colorize_bgr(
        self,
        img_bgr: np.ndarray,
        *,
        model: str = "ddcolor",
        input_size: int = 768,
        chroma: float = 1.35,
        cool: float = 0.22,
        render_factor: int = 30,
        deyellow: bool = True,
        autocontrast: bool = True,
        crop_page: bool = False,
    ) -> np.ndarray:
        crop, box = crop_photo_region(img_bgr, enabled=crop_page)
        prepared = prepare_input(crop, deyellow=deyellow, autocontrast=autocontrast)

        if model in ("ddcolor", "ddcolor-modelscope"):
            pipe = self.get_ddcolor("modelscope", input_size)
            out = pipe.process(prepared)
        elif model == "ddcolor-artistic":
            pipe = self.get_ddcolor("artistic", input_size)
            out = pipe.process(prepared)
        elif model in ("deoldify", "deoldify-stable"):
            out = self._colorize_deoldify_bgr(prepared, artistic=False, render_factor=render_factor)
        elif model == "deoldify-artistic":
            out = self._colorize_deoldify_bgr(prepared, artistic=True, render_factor=render_factor)
        else:
            raise ValueError(f"未知模型: {model}")

        out = adjust_color(out, chroma=chroma, cool=cool)
        if crop_page and box != (0, img_bgr.shape[0], 0, img_bgr.shape[1]):
            return paste_photo_region(img_bgr, out, box)
        return out

    def _colorize_deoldify_bgr(self, img_bgr, artistic: bool, render_factor: int) -> np.ndarray:
        colorizer = self.get_deoldify(artistic=artistic, render_factor=render_factor)
        tmp = _TEMP / f"_deo_{uuid.uuid4().hex}.jpg"
        imwrite_unicode(str(tmp), img_bgr)

        def _open_prepared(self, path):
            local = imread_unicode(str(path))
            return Image.fromarray(cv2.cvtColor(local, cv2.COLOR_BGR2RGB))

        colorizer._open_pil_image = types.MethodType(_open_prepared, colorizer)
        img_color = colorizer.get_transformed_image(path=str(tmp), render_factor=render_factor)
        tmp.unlink(missing_ok=True)
        arr = np.array(img_color)
        if arr.shape[0] > 20:
            arr = arr[:-20, :, :]
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def colorize_video(
        self,
        video_path: str,
        *,
        model: str = "ddcolor",
        input_size: int = 768,
        chroma: float = 1.35,
        cool: float = 0.22,
        render_factor: int = 21,
        deyellow: bool = True,
        autocontrast: bool = True,
        frame_stride: int = 1,
        max_frames: int = 0,
        output_path: str | None = None,
        progress=None,
    ) -> str:
        """Colorize video frame-by-frame. Returns output mp4 path."""
        video_path = str(video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(video_path)

        work = _TEMP / f"vid_{uuid.uuid4().hex}"
        bw_dir = work / "bw"
        color_dir = work / "color"
        bw_dir.mkdir(parents=True)
        color_dir.mkdir(parents=True)
        src_copy = work / ("input" + Path(video_path).suffix.lower())
        shutil.copy2(video_path, src_copy)

        # Extract frames
        if progress is not None:
            progress(0.02, desc="抽取视频帧...")
        pattern = str(bw_dir / "%06d.jpg")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src_copy),
                "-q:v", "2", pattern,
                "-hide_banner", "-loglevel", "error",
            ],
            check=True,
        )

        frames = sorted(bw_dir.glob("*.jpg"))
        if not frames:
            raise RuntimeError("未能从视频抽出帧，请检查文件是否损坏")

        if frame_stride < 1:
            frame_stride = 1
        selected = frames[::frame_stride]
        if max_frames and max_frames > 0:
            selected = selected[:max_frames]

        # Get fps
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1",
                str(src_copy),
            ],
            capture_output=True, text=True, check=False,
        )
        fps_txt = (probe.stdout or "25/1").strip()
        try:
            if "/" in fps_txt:
                a, b = fps_txt.split("/", 1)
                fps = float(a) / float(b)
            else:
                fps = float(fps_txt)
        except Exception:
            fps = 25.0
        out_fps = max(fps / frame_stride, 1.0)

        n = len(selected)
        for i, fp in enumerate(selected):
            if progress is not None:
                progress(0.05 + 0.85 * (i / max(n, 1)), desc=f"上色帧 {i+1}/{n}")
            img = imread_unicode(str(fp))
            out = self.colorize_bgr(
                img,
                model=model if model != "deoldify-video" else "deoldify",
                input_size=input_size,
                chroma=chroma,
                cool=cool,
                render_factor=render_factor,
                deyellow=deyellow,
                autocontrast=autocontrast,
                crop_page=False,
            )
            imwrite_unicode(str(color_dir / f"{i+1:06d}.jpg"), out)

        if progress is not None:
            progress(0.92, desc="合成视频...")

        no_audio = work / "colorized_no_audio.mp4"
        result = _resolve_video_output(video_path, output_path)
        result.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(out_fps),
                "-i", str(color_dir / "%06d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(no_audio),
                "-hide_banner", "-loglevel", "error",
            ],
            check=True,
        )

        # Try mux original audio
        mux = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(no_audio),
                "-i", str(src_copy),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(result),
                "-hide_banner", "-loglevel", "error",
            ],
            check=False,
        )
        if mux.returncode != 0 or not result.exists():
            shutil.copy2(no_audio, result)

        if progress is not None:
            progress(1.0, desc="完成")

        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass
        return str(result)


def resolve_video_output(video_path: str, output_path: str | None = None) -> Path:
    """解析视频输出路径：留空→results；填文件夹→文件夹内；填 .mp4→该文件。"""
    stem = Path(video_path).stem
    default_name = f"{stem}AI上色.mp4"
    raw = (output_path or "").strip().strip('"')
    if not raw:
        return RESULTS / default_name
    p = Path(raw)
    # 已存在的目录，或路径看起来像目录（无后缀）
    if p.is_dir() or (not p.suffix and not p.exists()):
        p.mkdir(parents=True, exist_ok=True)
        return p / default_name
    if p.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        p.mkdir(parents=True, exist_ok=True)
        return p / default_name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# 兼容 colorize_video 内部旧名
_resolve_video_output = resolve_video_output


ENGINE = ColorizeEngine()
