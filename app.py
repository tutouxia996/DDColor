"""Gradio web UI for image / video colorization.

启动:
  venv\\Scripts\\activate
  python app.py
"""

from __future__ import annotations

import os
import traceback
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from colorize_service import ENGINE, imread_unicode, imwrite_unicode, mean_chroma

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_CHOICES = [
    "ddcolor",
    "ddcolor-artistic",
    "deoldify",
    "deoldify-artistic",
]
MODEL_LABELS = {
    "ddcolor": "DDColor（推荐，更鲜艳）",
    "ddcolor-artistic": "DDColor Artistic（更淡）",
    "deoldify": "DeOldify Stable",
    "deoldify-artistic": "DeOldify Artistic",
}


def _bgr_from_gradio(img) -> np.ndarray:
    """Gradio Image may return RGB ndarray or path."""
    if img is None:
        raise ValueError("请先选择或上传一张图片")
    if isinstance(img, str):
        from colorize_service import imread_unicode

        return imread_unicode(img)
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("无法识别的图片格式")
    # Gradio 默认 RGB
    return cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)


def run_image(
    image,
    model,
    input_size,
    chroma,
    cool,
    render_factor,
    deyellow,
    autocontrast,
    crop_page,
):
    try:
        img_bgr = _bgr_from_gradio(image)
        out = ENGINE.colorize_bgr(
            img_bgr,
            model=model,
            input_size=int(input_size),
            chroma=float(chroma),
            cool=float(cool),
            render_factor=int(render_factor),
            deyellow=bool(deyellow),
            autocontrast=bool(autocontrast),
            crop_page=bool(crop_page),
        )
        out_path = RESULTS / f"ui_{uuid.uuid4().hex[:10]}.jpg"
        imwrite_unicode(str(out_path), out)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        info = f"完成 | 色度≈{mean_chroma(out):.1f} | 已保存: {out_path}"
        return rgb, info
    except Exception as e:
        traceback.print_exc()
        return None, f"失败: {e}"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _file_path_from_upload(item) -> str | None:
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("path") or item.get("name")
    name = getattr(item, "name", None)
    return str(name) if name else None


def _collect_batch_paths(files, folder: str) -> tuple[list[Path], Path | None]:
    """Return (image paths, input folder used for naming)."""
    paths: list[Path] = []
    seen: set[str] = set()
    src_folder: Path | None = None

    def _add(p: Path):
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        if p.suffix.lower() in IMAGE_EXTS:
            seen.add(key)
            paths.append(p)

    folder = (folder or "").strip().strip('"')
    if folder:
        root = Path(folder)
        if not root.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder}")
        if root.is_file():
            src_folder = root.parent
            _add(root)
        else:
            src_folder = root
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    _add(p)

    if files:
        items = files if isinstance(files, list) else [files]
        for item in items:
            fp = _file_path_from_upload(item)
            if fp:
                _add(Path(fp))

    if src_folder is None and paths:
        src_folder = paths[0].parent
    return paths, src_folder


def _resolve_output_dir(src_folder: Path | None, output_parent: str) -> Path:
    name = src_folder.name if src_folder else "图片"
    out_name = name if name.endswith("AI上色") else f"{name}AI上色"

    output_parent = (output_parent or "").strip().strip('"')
    if output_parent:
        parent = Path(output_parent)
    elif src_folder is not None:
        parent = src_folder.parent
    else:
        parent = RESULTS
    return parent / out_name


def preview_batch_output(folder: str, output_parent: str) -> str:
    folder = (folder or "").strip().strip('"')
    src = Path(folder) if folder else None
    if src and src.is_file():
        src = src.parent
    try:
        return str(_resolve_output_dir(src if src and src.exists() else None, output_parent))
    except Exception:
        return ""


def run_image_batch(
    files,
    folder,
    output_parent,
    model,
    input_size,
    chroma,
    cool,
    render_factor,
    deyellow,
    autocontrast,
    crop_page,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        paths, src_folder = _collect_batch_paths(files, folder)
        if not paths:
            return [], "请先多选图片，或填写图片文件夹路径"

        out_dir = _resolve_output_dir(src_folder, output_parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        previews = []
        ok = 0
        lines = []
        n = len(paths)
        for i, src in enumerate(paths):
            progress((i + 1) / n, desc=f"上色 {i + 1}/{n}: {src.name}")
            try:
                img_bgr = imread_unicode(str(src))
                out = ENGINE.colorize_bgr(
                    img_bgr,
                    model=model,
                    input_size=int(input_size),
                    chroma=float(chroma),
                    cool=float(cool),
                    render_factor=int(render_factor),
                    deyellow=bool(deyellow),
                    autocontrast=bool(autocontrast),
                    crop_page=bool(crop_page),
                )
                dest = out_dir / f"{src.stem}.jpg"
                if src_folder and src_folder.exists() and src_folder.is_dir():
                    try:
                        dest = (out_dir / src.relative_to(src_folder)).with_suffix(".jpg")
                    except ValueError:
                        pass
                dest.parent.mkdir(parents=True, exist_ok=True)
                imwrite_unicode(str(dest), out)
                previews.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                ok += 1
                lines.append(f"[OK c={mean_chroma(out):.1f}] {src.name}")
            except Exception as e:
                lines.append(f"[FAIL] {src.name}: {e}")

        info = f"完成：{ok}/{n} | 保存目录: {out_dir}\n" + "\n".join(lines)
        return previews, info
    except Exception as e:
        traceback.print_exc()
        return [], f"失败: {e}"


def run_video(
    video,
    model,
    input_size,
    chroma,
    cool,
    render_factor,
    deyellow,
    autocontrast,
    frame_stride,
    max_frames,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        if video is None:
            return None, "请先上传黑白视频"
        # Gradio may give path str or dict
        if isinstance(video, dict):
            video_path = video.get("path") or video.get("name")
        else:
            video_path = video
        if not video_path or not os.path.isfile(str(video_path)):
            return None, f"无效视频路径: {video_path}"

        out_path = ENGINE.colorize_video(
            str(video_path),
            model=model,
            input_size=int(input_size),
            chroma=float(chroma),
            cool=float(cool),
            render_factor=int(render_factor),
            deyellow=bool(deyellow),
            autocontrast=bool(autocontrast),
            frame_stride=int(frame_stride),
            max_frames=int(max_frames) if max_frames else 0,
            progress=progress,
        )
        return out_path, f"完成 | 已保存: {out_path}"
    except Exception as e:
        traceback.print_exc()
        return None, f"失败: {e}"


def build_ui():
    with gr.Blocks(title="老照片 / 视频 AI 上色", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 老照片 / 黑白视频 AI 上色
            支持 **DDColor** 与 **DeOldify**。默认参数为上一档：`input-size=768`，`chroma=1.35`。
            """
        )

        with gr.Row():
            model = gr.Dropdown(
                choices=MODEL_CHOICES,
                value="ddcolor",
                label="模型（ddcolor / ddcolor-artistic / deoldify / deoldify-artistic）",
            )
            input_size = gr.Slider(512, 1024, value=768, step=128, label="input-size（越大越清晰、越慢）")
            chroma = gr.Slider(0.8, 1.8, value=1.35, step=0.05, label="chroma 色彩浓度（大片天空图可先试 1.15～1.25）")
            cool = gr.Slider(0.0, 0.8, value=0.22, step=0.02, label="cool 压红强度")

        with gr.Accordion("更多参数", open=False):
            with gr.Row():
                render_factor = gr.Slider(12, 40, value=30, step=1, label="DeOldify render_factor")
                deyellow = gr.Checkbox(value=True, label="去黄（发黄原片建议开）")
                autocontrast = gr.Checkbox(value=True, label="自动对比度")
                crop_page = gr.Checkbox(value=False, label="裁相纸页边（仅图片，有白页边时用）")

        with gr.Tabs():
            with gr.Tab("图片上色"):
                with gr.Tabs():
                    with gr.Tab("单张"):
                        with gr.Row():
                            with gr.Column():
                                in_img = gr.Image(label="上传黑白/发黄老照片", type="numpy")
                                btn_img = gr.Button("开始上色", variant="primary")
                            with gr.Column():
                                out_img = gr.Image(label="上色结果", type="numpy")
                                img_log = gr.Textbox(label="状态", lines=2)
                        btn_img.click(
                            run_image,
                            inputs=[
                                in_img, model, input_size, chroma, cool, render_factor,
                                deyellow, autocontrast, crop_page,
                            ],
                            outputs=[out_img, img_log],
                        )
                    with gr.Tab("批量"):
                        gr.Markdown(
                            "可一次多选图片，或填写输入文件夹。输出文件夹名固定为：**输入文件夹名 + AI上色**。"
                        )
                        with gr.Row():
                            with gr.Column():
                                in_files = gr.File(
                                    label="多选图片",
                                    file_count="multiple",
                                    file_types=["image"],
                                )
                                in_folder = gr.Textbox(
                                    label="输入文件夹路径",
                                    placeholder=r"例如 D:\photos\京张路工",
                                )
                                out_parent = gr.Textbox(
                                    label="输出位置（填父文件夹；留空则和输入文件夹同级）",
                                    placeholder=r"例如 D:\photos",
                                )
                                out_preview = gr.Textbox(
                                    label="将保存到",
                                    interactive=False,
                                )
                                btn_batch = gr.Button("开始批量上色", variant="primary")
                            with gr.Column():
                                out_gallery = gr.Gallery(
                                    label="上色结果预览",
                                    columns=3,
                                    height=520,
                                    object_fit="contain",
                                )
                                batch_log = gr.Textbox(label="状态", lines=12)
                        in_folder.change(
                            preview_batch_output,
                            inputs=[in_folder, out_parent],
                            outputs=[out_preview],
                        )
                        out_parent.change(
                            preview_batch_output,
                            inputs=[in_folder, out_parent],
                            outputs=[out_preview],
                        )
                        btn_batch.click(
                            run_image_batch,
                            inputs=[
                                in_files, in_folder, out_parent, model, input_size, chroma, cool,
                                render_factor, deyellow, autocontrast, crop_page,
                            ],
                            outputs=[out_gallery, batch_log],
                        )

            with gr.Tab("视频上色"):
                gr.Markdown(
                    "逐帧上色，时间较长。可先把 **隔帧** 调大（如 2/3）做预览，满意后再设为 1 精跑。"
                )
                with gr.Row():
                    with gr.Column():
                        in_vid = gr.Video(label="上传黑白视频")
                        frame_stride = gr.Slider(1, 5, value=1, step=1, label="隔帧（1=每帧都上色）")
                        max_frames = gr.Slider(0, 500, value=0, step=10, label="最多处理帧数（0=不限制）")
                        btn_vid = gr.Button("开始上色视频", variant="primary")
                    with gr.Column():
                        out_vid = gr.Video(label="上色结果")
                        vid_log = gr.Textbox(label="状态", lines=3)
                btn_vid.click(
                    run_video,
                    inputs=[
                        in_vid, model, input_size, chroma, cool, render_factor,
                        deyellow, autocontrast, frame_stride, max_frames,
                    ],
                    outputs=[out_vid, vid_log],
                )

        gr.Markdown(
            f"结果默认保存到 `{RESULTS}`。命令行仍可用：`python zhixing.py`"
        )
    return demo


if __name__ == "__main__":
    # 避免公司代理 / 网络策略导致 Gradio 误判 localhost 不可用
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,0.0.0.0")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost,0.0.0.0")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    # 兼容旧 gradio_client：JSON schema 里 additionalProperties 可能是 bool
    try:
        from gradio_client import utils as _gc_utils

        _orig_get_type = _gc_utils.get_type

        def _safe_get_type(schema):
            if isinstance(schema, bool):
                return "Any"
            return _orig_get_type(schema)

        _gc_utils.get_type = _safe_get_type

        _orig_json = _gc_utils._json_schema_to_python_type

        def _safe_json(schema, defs=None):
            if isinstance(schema, bool):
                return "Any"
            return _orig_json(schema, defs)

        _gc_utils._json_schema_to_python_type = _safe_json
        _gc_utils.json_schema_to_python_type = lambda schema: _safe_json(
            schema, schema.get("$defs") if isinstance(schema, dict) else None
        )
    except Exception:
        pass

    os.chdir(ROOT)
    demo = build_ui()

    def _pick_port(start: int = 7860, tries: int = 20) -> int:
        import socket

        for port in range(start, start + tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise OSError(f"No free port in {start}-{start + tries - 1}")

    port = _pick_port(7860)
    print(f"Starting UI at http://127.0.0.1:{port} ...")
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True,
        show_error=True,
        share=False,
    )
