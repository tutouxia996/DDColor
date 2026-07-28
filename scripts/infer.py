#!/usr/bin/env python
"""
DDColor inference script.

Supports two modes:
1. Local weights: python scripts/infer.py --model_path path/to/model.pt --input ./images
2. Hugging Face:  python scripts/infer.py --model_name ddcolor_modelscope --input ./images
3. Single image:  python scripts/infer.py --model_path path/to/model.pt --input ./test.jpg --output ./result.jpg
"""

import os
import sys
import argparse
import traceback

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread 替代函数，支持中文路径（Windows 兼容）。

    关键：Python open() 支持 UTF-8 路径，而 cv2.imread / np.fromfile 底层
    使用 C fopen，在 Windows 上对非 ASCII 路径会失败。先用 open 读入 bytes，
    再用 cv2.imdecode 解码。
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
        if img is None:
            print(f"[WARN] imdecode 返回 None，文件可能是损坏的图片: {filepath}")
        return img
    except FileNotFoundError:
        print(f"[ERROR] 文件不存在: {filepath}")
        return None
    except Exception as e:
        print(f"[ERROR] 读取图片失败: {filepath}")
        traceback.print_exc()
        return None


def imwrite_unicode(filepath: str, img: np.ndarray, params=None) -> bool:
    """cv2.imwrite 替代函数，支持中文路径（Windows 兼容）。

    关键：ndarray.tofile() 底层也是 C fopen，Windows 上不支持非 ASCII 路径。
    先用 cv2.imencode 编码为 bytes，再用 Python open() 写入。
    """
    ext = os.path.splitext(filepath)[1].lower()  # 统一小写，兼容 .JPG/.PNG
    if not ext:
        ext = '.jpg'  # 无扩展名时默认 jpg
    try:
        success, buf = cv2.imencode(ext, img, params)
        if not success:
            print(f"[ERROR] imencode 失败: {filepath}")
            return False
        with open(filepath, 'wb') as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:
        print(f"[ERROR] 写入图片失败: {filepath}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="DDColor inference script")

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model_path', type=str,
        help='Path to the local model weights (.pt file)'
    )
    model_group.add_argument(
        '--model_name', type=str,
        help='Hugging Face model name (e.g., ddcolor_modelscope, ddcolor_paper, ddcolor_artistic, ddcolor_paper_tiny)'
    )

    # Common arguments
    parser.add_argument('--input', type=str, default='assets/test_images', help='Input image file or folder')
    parser.add_argument('--output', type=str, default='results', help='Output folder (or file for single image)')
    parser.add_argument('--input_size', type=int, default=512, help='Input size for the model')
    parser.add_argument('--model_size', type=str, default='large', choices=['tiny', 'large'],
                        help='DDColor model size (only used with --model_path)')

    args = parser.parse_args()

    # ===================== 核心修改：兼容单文件/文件夹 =====================
    single_image_mode = False
    input_file_path = ""
    output_file_path = args.output

    # 判断输入是文件还是文件夹
    if os.path.isfile(args.input):
        single_image_mode = True
        # 提取单文件的文件夹和文件名
        input_dir = os.path.dirname(args.input)
        input_filename = os.path.basename(args.input)
        # 处理输出：如果输出是文件夹，自动拼接文件名；如果是文件路径，直接使用
        if os.path.isdir(args.output) or not args.output.endswith(('.jpg', '.png', '.jpeg')):
            os.makedirs(args.output, exist_ok=True)
            output_file_path = os.path.join(args.output, input_filename)
        # 重置input为文件夹，file_list只包含当前文件
        file_list = [input_filename]
        args.input = input_dir
    else:
        # 原逻辑：文件夹输入
        file_list = os.listdir(args.input)
        file_list = [f for f in file_list if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))]
        assert len(file_list) > 0, "No images found in the input directory."
        file_list = sorted(file_list)
        os.makedirs(args.output, exist_ok=True)
    # =====================================================================

    print(f'Output path: {output_file_path if single_image_mode else args.output}')
    print(f'File list ({len(file_list)} files): {file_list[:5]}{"..." if len(file_list) > 5 else ""}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ===================== 模型加载 =====================
    print('Loading model...')
    try:
        if args.model_path:
            # Local weights mode
            model = build_ddcolor_model(
                DDColor,
                model_path=args.model_path,
                input_size=args.input_size,
                model_size=args.model_size,
                device=device,
            )
        else:
            # Hugging Face mode
            from huggingface_hub import PyTorchModelHubMixin

            class DDColorHF(DDColor, PyTorchModelHubMixin):
                def __init__(self, config=None, **kwargs):
                    if isinstance(config, dict):
                        kwargs = {**config, **kwargs}
                    super().__init__(**kwargs)

            model_name = args.model_name
            if not os.path.isdir(model_name):
                model_name = f"piddnad/{model_name}"

            model = DDColorHF.from_pretrained(model_name)
            model = model.to(device)
            model.eval()
    except Exception as e:
        print(f"[FATAL] 模型加载失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    print('Model loaded successfully.')
    colorizer = ColorizationPipeline(model, input_size=args.input_size, device=device)

    # ===================== 处理图片 =====================
    if single_image_mode:
        # 单文件模式：直接处理
        img_path = os.path.join(args.input, file_list[0])
        print(f'Reading: {img_path}')
        img = imread_unicode(img_path)
        if img is not None:
            print(f'Image shape: {img.shape}, processing...')
            try:
                image_out = colorizer.process(img)
                print(f'Writing: {output_file_path}')
                ok = imwrite_unicode(output_file_path, image_out)
                if ok:
                    print(f"✅ Single image processed! Saved to: {output_file_path}")
                else:
                    print(f"❌ Failed to write: {output_file_path}")
            except Exception as e:
                print(f"[ERROR] 图片处理失败: {e}")
                traceback.print_exc()
        else:
            print(f"❌ Failed to read {img_path}")
    else:
        # 文件夹模式：批量处理
        for file_name in tqdm(file_list):
            img_path = os.path.join(args.input, file_name)
            img = imread_unicode(img_path)
            if img is not None:
                try:
                    image_out = colorizer.process(img)
                    out_path = os.path.join(args.output, file_name)
                    imwrite_unicode(out_path, image_out)
                except Exception as e:
                    print(f"\n[ERROR] 处理 {file_name} 失败: {e}")
                    traceback.print_exc()
            else:
                print(f"\n❌ Failed to read {img_path}")


if __name__ == '__main__':
    main()