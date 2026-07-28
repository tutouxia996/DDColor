import os
import cv2
import numpy as np
import torch
import functools
import fastai.basic_train

# ========== 强制所有临时文件存到D盘 ==========
import tempfile
# 1. 新建D盘临时文件夹
temp_dir = "D:\AI_Temp"
os.makedirs(temp_dir, exist_ok=True)
# 2. 让Python临时文件存到D盘
tempfile.tempdir = temp_dir
# 3. 让OpenCV临时文件存到D盘
os.environ['OPENCV_TEMP_DIR'] = temp_dir
# 4. 让PyTorch缓存存到D盘
os.environ['TORCH_HOME'] = os.path.join(temp_dir, "torch_cache")
os.environ['FASTAI_HOME'] = os.path.join(temp_dir, "fastai_cache")

# ===================== 兼容低版本PyTorch 2.1.0 =====================
if not hasattr(torch.serialization, 'add_safe_globals'):
    def add_safe_globals(globals_list):
        pass


    torch.serialization.add_safe_globals = add_safe_globals


# ===================== 检测并启用GPU（4060专属） =====================
def check_cuda():
    if not torch.cuda.is_available():
        print("❌ 未检测到可用GPU，自动切换到CPU运行")
        return torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"✅ 成功启用GPU：{torch.cuda.get_device_name(0)}")
        return device


DEVICE = check_cuda()

# ===================== 重写torch.load，适配GPU+低版本 =====================
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if 'weights_only' in kwargs:
        del kwargs['weights_only']
    if DEVICE.type == "cuda":
        kwargs['map_location'] = DEVICE
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

# ===================== 基础配置 =====================
import warnings

warnings.filterwarnings("ignore")
os.environ['DEOLDIFY_MODEL_DIR'] = os.path.join(os.getcwd(), 'models')

try:
    from deoldify.visualize import get_image_colorizer
except ImportError as e:
    print(f"❌ 依赖缺失：{e}")
    print("💡 执行：pip install fastprogress fastai==2.7.10 deoldify -i https://pypi.tuna.tsinghua.edu.cn/simple")
    exit(1)

# 创建文件夹
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===================== 检查权重 =====================
MODEL_PATH = os.path.join("models", "ColorizeStable_gen.pth")
if not os.path.exists(MODEL_PATH):
    print(f"❌ 未找到模型权重：{MODEL_PATH}")
    print("💡 请先运行 download_deoldify_model.py 下载模型")
    exit(1)

# ===================== 初始化模型（核心修复：正确指定GPU） =====================
print("ℹ️ 正在加载DeOldify模型...")
# 修复：DeOldify的colorizer通过learn属性访问模型，且无需手动迁移（已通过map_location加载到GPU）
colorizer = get_image_colorizer(artistic=False)
colorizer._device = DEVICE  # 仅指定设备属性，无需手动迁移模型（已通过torch.load的map_location加载到GPU）
colorizer.render_factor = 35


# ===================== 批量处理核心函数 =====================
def process_single_image(img_path, colorizer):
    img_name = os.path.basename(img_path)
    relative_path = os.path.relpath(img_path, "test_images")
    output_dir = os.path.join("results", os.path.dirname(relative_path))
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_IMAGE = os.path.join(output_dir, f"deoldify_{img_name}")

    try:
        # 上色（DeOldify会自动使用指定的_device）
        img_color = colorizer.get_transformed_image(
            path=img_path,
            render_factor=35
        )

        # 裁剪底部水印
        img_array = np.array(img_color)
        if img_array.shape[0] > 20:
            img_array = img_array[:-20, :, :]

        # 转换通道并保存
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(OUTPUT_IMAGE, img_bgr)

        print(f"✅ 处理完成：{img_path}")
        print(f"   结果保存：{os.path.abspath(OUTPUT_IMAGE)}")
        return True
    except Exception as e:
        print(f"\n❌ 处理失败 {img_path}：{str(e)}")
        if "out of memory" in str(e).lower():
            print("💡 把render_factor=35改成20即可解决显存不足")
        return False


def batch_process():
    supported_formats = (".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".bmp")
    img_paths = []
    for root, dirs, files in os.walk("test_images"):
        for file in files:
            if file.endswith(supported_formats):
                img_paths.append(os.path.join(root, file))

    if not img_paths:
        print(f"❌ 在test_images及子文件夹中未找到图片（支持格式：{supported_formats}）")
        exit(1)

    print(f"\n📌 共找到 {len(img_paths)} 张图片，开始批量上色...")
    success_count = 0
    for img_path in img_paths:
        if process_single_image(img_path, colorizer):
            success_count += 1

    print(f"\n🎉 批量处理结束！成功 {success_count}/{len(img_paths)} 张")
    print(f"📁 所有结果保存在：{os.path.abspath('results')}")


# ===================== 执行 =====================
if __name__ == "__main__":
    batch_process()