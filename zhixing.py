"""Batch colorize old photos with DeOldify and/or DDColor.

针对发黄相纸册页（如京张路工写真）的推荐命令：
  python zhixing.py
  # 默认即为上一档：chroma=1.35, input-size=768
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ========== Force temp / caches onto D: ==========
temp_dir = r"D:\AI_Temp"
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir
os.environ["OPENCV_TEMP_DIR"] = temp_dir
os.environ["TORCH_HOME"] = os.path.join(temp_dir, "torch_cache")
os.environ["FASTAI_HOME"] = os.path.join(temp_dir, "fastai_cache")
os.environ.setdefault("HF_HOME", os.path.join(temp_dir, "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
os.environ["DEOLDIFY_MODEL_DIR"] = str(MODELS)

if not hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals = lambda globals_list: None  # type: ignore


def check_cuda():
    if not torch.cuda.is_available():
        print("未检测到可用 GPU，使用 CPU")
        return torch.device("cpu")
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


DEVICE = check_cuda()
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.pop("weights_only", None)
    if DEVICE.type == "cuda":
        kwargs.setdefault("map_location", DEVICE)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
warnings.filterwarnings("ignore")


def imread_unicode(filepath: str):
    with open(filepath, "rb") as f:
        data = f.read()
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {filepath}")
    return img


def imwrite_unicode(filepath: str, img) -> None:
    ext = os.path.splitext(filepath)[1].lower() or ".jpg"
    success, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise IOError(f"编码失败: {filepath}")
    with open(filepath, "wb") as f:
        f.write(buf.tobytes())


def collect_images(input_dir: Path):
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if Path(name).suffix.lower() in supported:
                paths.append(Path(root) / name)
    return sorted(paths)


def resolve_ddcolor_weight(kind: str) -> Path:
    if kind == "modelscope":
        candidates = [
            MODELS / "ddcolor_modelscope.bin",
            MODELS / "ddcolor_modelscope.pt",
            ROOT / "modelscope" / "damo" / "cv_ddcolor_image-colorization" / "pytorch_model.pt",
        ]
    else:
        candidates = [
            MODELS / "ddcolor_artistic.bin",
            MODELS / "ddcolor_artistic.pt",
        ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 100_000_000:
            return p
    raise FileNotFoundError(
        f"找不到 DDColor ({kind}) 权重。请先运行: python download_models.py\n"
        f"已尝试: {', '.join(str(c) for c in candidates)}"
    )


def load_deoldify(artistic: bool, render_factor: int):
    try:
        from deoldify.visualize import get_image_colorizer
    except ImportError as e:
        print(f"DeOldify 依赖缺失: {e}")
        sys.exit(1)

    need = "ColorizeArtistic_gen.pth" if artistic else "ColorizeStable_gen.pth"
    weight = MODELS / need
    if not weight.exists():
        print(f"缺少 DeOldify 权重: {weight}")
        sys.exit(1)

    print(f"加载 DeOldify ({'Artistic' if artistic else 'Stable'}) ...")
    colorizer = get_image_colorizer(root_folder=ROOT, artistic=artistic, render_factor=render_factor)
    colorizer._device = DEVICE
    colorizer.render_factor = render_factor
    return colorizer


def load_ddcolor(kind: str, input_size: int):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Ensure latest pipeline.py is picked up
    for mod in list(sys.modules):
        if mod == "ddcolor" or mod.startswith("ddcolor."):
            del sys.modules[mod]
    from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

    weight = resolve_ddcolor_weight(kind)
    print(f"加载 DDColor ({kind}) @ {input_size} <- {weight.name}")
    model = build_ddcolor_model(
        DDColor,
        model_path=str(weight),
        input_size=input_size,
        model_size="large",
        device=DEVICE,
    )
    return ColorizationPipeline(model, input_size=input_size, device=DEVICE)


def out_path_for(img_path: Path, input_root: Path, output_root: Path, prefix: str) -> Path:
    rel = img_path.relative_to(input_root)
    dest_dir = output_root / rel.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{prefix}_{img_path.name}"


def mean_chroma(img_bgr: np.ndarray) -> float:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.sqrt((lab[:, :, 1] - 128) ** 2 + (lab[:, :, 2] - 128) ** 2).mean())


def crop_photo_region(img_bgr: np.ndarray, enabled: bool = True):
    """仅在确有白页边时裁切；无页边则原样返回，避免贴回灰边框。"""
    h, w = img_bgr.shape[:2]
    if not enabled:
        return img_bgr, (0, h, 0, w)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    white = gray >= 245
    white_ratio = float(white.mean())
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
    pad_y = max(2, (y1 - y0) // 200)
    pad_x = max(2, (x1 - x0) // 200)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)

    if (y1 - y0) > h * 0.97 and (x1 - x0) > w * 0.97:
        return img_bgr, (0, h, 0, w)
    if (y1 - y0) < h * 0.45 or (x1 - x0) < w * 0.45:
        return img_bgr, (0, h, 0, w)
    return img_bgr[y0:y1, x0:x1], (y0, y1, x0, x1)


def paste_photo_region(full_bgr: np.ndarray, crop_bgr: np.ndarray, box) -> np.ndarray:
    """贴回时：页边用浅色纸底，不再用整图灰度，避免出现黑白相框。"""
    y0, y1, x0, x1 = box
    out = np.full_like(full_bgr, 245)
    ch, cw = crop_bgr.shape[:2]
    if (ch, cw) != (y1 - y0, x1 - x0):
        crop_bgr = cv2.resize(crop_bgr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    out[y0:y1, x0:x1] = crop_bgr
    return out


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
    yellow = np.clip((b - 128.0) / 16.0, 0.0, 1.0)
    greenish = np.clip((128.0 - a) / 14.0, 0.0, 1.0) * np.clip((b - 120.0) / 14.0, 0.0, 1.0)
    cast = np.clip(yellow * 0.9 + greenish * 1.2, 0.0, 1.0)
    blotch = (sky_base * cast > 0.22).astype(np.uint8) * 255
    k = max(5, (min(h, w) // 80) | 1)
    blotch = cv2.morphologyEx(blotch, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    blotch = cv2.dilate(blotch, np.ones((k, k), np.uint8), iterations=1)
    blotch = cv2.GaussianBlur(blotch.astype(np.float32) / 255.0, (0, 0), max(3.0, k / 2.0))
    sky = np.clip(sky_base * cast * 0.55 + blotch * sky_base * 0.85, 0.0, 1.0)
    a = a * (1.0 - sky * 0.95) + 128.0 * (sky * 0.95)
    b = b * (1.0 - sky * 0.96) + 118.0 * (sky * 0.96)
    hi = np.clip((L - 155.0) / 75.0, 0.0, 1.0) * sky_base
    b = b - hi * np.clip(b - 128.0, 0.0, None) * 0.85
    a = a + hi * np.clip(120.0 - a, 0.0, None) * 0.45
    return a, b


def adjust_color(img_bgr: np.ndarray, chroma: float = 1.35, cool: float = 0.25) -> np.ndarray:
    """增强颜色、压黄、去红蓝边缘重影。"""
    chroma = float(np.clip(chroma, 0.5, 2.2))
    cool = float(np.clip(cool, 0.0, 1.0))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    cur = float(np.sqrt((a - 128) ** 2 + (b - 128) ** 2).mean())
    auto_boost = 1.0
    if cur < 5:
        auto_boost = 1.55
    elif cur < 9:
        auto_boost = 1.25
    chroma_eff = chroma * auto_boost

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
        red_excess = np.clip(a - 145.0, 0.0, None)
        a -= red_excess * (0.5 * cool)

    L8 = np.clip(L, 0, 255).astype(np.uint8)
    edges = cv2.Canny(L8, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (5, 5), 0)
    a_s = cv2.bilateralFilter(a.astype(np.float32), 7, 18, 7)
    b_s = cv2.bilateralFilter(b.astype(np.float32), 7, 18, 7)
    a = a * (1.0 - edges * 0.7) + a_s * (edges * 0.7)
    b = b * (1.0 - edges * 0.7) + b_s * (edges * 0.7)

    out = cv2.cvtColor(
        cv2.merge(
            [
                np.clip(L, 0, 255).astype(np.uint8),
                np.clip(a, 0, 255).astype(np.uint8),
                np.clip(b, 0, 255).astype(np.uint8),
            ]
        ),
        cv2.COLOR_LAB2BGR,
    )
    return out


def colorize_with_deoldify(colorizer, img_bgr: np.ndarray, render_factor: int) -> np.ndarray:
    """DeOldify 接受路径；这里写临时文件。"""
    tmp = Path(temp_dir) / "_deoldify_in.jpg"
    imwrite_unicode(str(tmp), img_bgr)

    def _open_prepared(self, path):
        local = imread_unicode(str(path))
        return Image.fromarray(cv2.cvtColor(local, cv2.COLOR_BGR2RGB))

    colorizer._open_pil_image = types.MethodType(_open_prepared, colorizer)
    img_color = colorizer.get_transformed_image(path=str(tmp), render_factor=render_factor)
    arr = np.array(img_color)
    if arr.shape[0] > 20:
        arr = arr[:-20, :, :]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def process_one(
    img_path: Path,
    dest: Path,
    *,
    dd_pipeline,
    deoldify_colorizer,
    render_factor: int,
    chroma: float,
    cool: float,
    deyellow: bool,
    autocontrast: bool,
    crop_page: bool,
    rescue_chroma: float,
):
    full = imread_unicode(str(img_path))
    crop, box = crop_photo_region(full, enabled=crop_page)
    prepared = prepare_input(crop, deyellow=deyellow, autocontrast=autocontrast)

    out = dd_pipeline.process(prepared)
    out = adjust_color(out, chroma=chroma, cool=cool)
    c1 = mean_chroma(out)

    rescued = False
    if c1 < rescue_chroma and deoldify_colorizer is not None:
        try:
            d_out = colorize_with_deoldify(deoldify_colorizer, prepared, render_factor)
            d_out = adjust_color(d_out, chroma=max(chroma, 1.25), cool=cool)
            c2 = mean_chroma(d_out)
            if c2 > c1 * 1.15:
                out = d_out
                rescued = True
                c1 = c2
        except Exception as e:
            print(f"  [rescue-skip] {e}")

    # 贴回整页
    if crop_page and box != (0, full.shape[0], 0, full.shape[1]):
        # 页边中性灰
        full_gray = neutralize_yellow(full)
        final = paste_photo_region(full_gray, out, box)
    else:
        final = out

    imwrite_unicode(str(dest), final)
    tag = "rescued" if rescued else "ok"
    print(f"[OK:{tag} c={c1:.1f}] {img_path.name} -> {dest.name}")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="老照片上色：DeOldify / DDColor")
    p.add_argument(
        "--model",
        default="ddcolor",
        choices=[
            "deoldify",
            "deoldify-stable",
            "deoldify-artistic",
            "ddcolor",
            "ddcolor-modelscope",
            "ddcolor-artistic",
            "both",
        ],
        help="默认 ddcolor(=modelscope，更鲜艳)。artistic 更淡、易出灰片",
    )
    p.add_argument("--input", default="test_images", help="输入目录")
    p.add_argument("--output", default="results", help="输出目录")
    p.add_argument("--render-factor", type=int, default=30, help="DeOldify 强度，补救灰片时用")
    p.add_argument("--input-size", type=int, default=768, help="DDColor 推理边长，越大越清晰、重影越少")
    p.add_argument("--chroma", type=float, default=1.35, help="色彩浓度（默认 1.35，上一档；别盲目加到 1.5）")
    p.add_argument("--cool", type=float, default=0.22, help="轻压红斑")
    p.add_argument("--rescue-chroma", type=float, default=6.0, help="低于此色度则用 DeOldify 补救")
    p.add_argument("--no-deyellow", action="store_true")
    p.add_argument("--no-autocontrast", action="store_true")
    p.add_argument("--no-crop-page", action="store_true", help="关闭相纸页芯裁切")
    p.add_argument("--no-rescue", action="store_true", help="关闭灰片自动 DeOldify 补救")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    return p.parse_args()


def main():
    args = parse_args()
    input_root = (ROOT / args.input).resolve()
    output_root = (ROOT / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    img_paths = collect_images(input_root)
    if args.limit > 0:
        img_paths = img_paths[: args.limit]
    if not img_paths:
        print(f"在 {input_root} 下没有找到图片")
        sys.exit(1)

    deyellow = not args.no_deyellow
    autocontrast = not args.no_autocontrast
    crop_page = not args.no_crop_page
    use_rescue = not args.no_rescue

    print(
        f"共 {len(img_paths)} 张 | model={args.model} size={args.input_size} "
        f"chroma={args.chroma} | 去黄={deyellow} 裁页={crop_page} 补救={use_rescue}"
    )

    # 主模型：ddcolor / modelscope 更鲜艳；artistic 保留可选
    want_dd = args.model in (
        "ddcolor",
        "ddcolor-modelscope",
        "ddcolor-artistic",
        "both",
    )
    want_deo = args.model in (
        "deoldify",
        "deoldify-stable",
        "deoldify-artistic",
        "both",
    ) or use_rescue

    dd_kind = "artistic" if args.model == "ddcolor-artistic" else "modelscope"
    dd_pipeline = load_ddcolor(dd_kind, args.input_size) if want_dd else None

    deo_artistic = args.model == "deoldify-artistic"
    deoldify_colorizer = None
    if want_deo:
        # 补救默认用 Stable，更稳
        deoldify_colorizer = load_deoldify(
            artistic=deo_artistic if args.model.startswith("deoldify") else False,
            render_factor=args.render_factor,
        )

    ok = 0
    if args.model.startswith("deoldify") and not args.model.startswith("ddcolor"):
        # 纯 DeOldify 模式
        for img_path in img_paths:
            dest = out_path_for(img_path, input_root, output_root, "deoldify")
            try:
                full = imread_unicode(str(img_path))
                crop, box = crop_photo_region(full, enabled=crop_page)
                prepared = prepare_input(crop, deyellow=deyellow, autocontrast=autocontrast)
                out = colorize_with_deoldify(deoldify_colorizer, prepared, args.render_factor)
                out = adjust_color(out, chroma=args.chroma, cool=args.cool)
                if crop_page:
                    final = paste_photo_region(neutralize_yellow(full), out, box)
                else:
                    final = out
                imwrite_unicode(str(dest), final)
                print(f"[OK c={mean_chroma(out):.1f}] {img_path.name}")
                ok += 1
            except Exception as e:
                print(f"[FAIL] {img_path.name}: {e}")
    else:
        prefix = "ddcolor_artistic" if dd_kind == "artistic" else "ddcolor"
        for img_path in img_paths:
            dest = out_path_for(img_path, input_root, output_root, prefix)
            try:
                if process_one(
                    img_path,
                    dest,
                    dd_pipeline=dd_pipeline,
                    deoldify_colorizer=deoldify_colorizer if use_rescue else None,
                    render_factor=args.render_factor,
                    chroma=args.chroma,
                    cool=args.cool,
                    deyellow=deyellow,
                    autocontrast=autocontrast,
                    crop_page=crop_page,
                    rescue_chroma=args.rescue_chroma,
                ):
                    ok += 1
            except Exception as e:
                print(f"[FAIL] {img_path.name}: {e}")
                if "out of memory" in str(e).lower():
                    print("提示: 把 --input-size 降到 512")

    print(f"\n完成：{ok}/{len(img_paths)}")
    print(f"结果目录：{output_root}")
    print("当前默认：chroma=1.35 input-size=768（上一档）")


if __name__ == "__main__":
    main()
