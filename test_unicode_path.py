#!/usr/bin/env python
"""测试中文路径图片读写是否正常。

用法: python test_unicode_path.py <中文图片路径>
结果: 会在同一目录生成 test_output_中文原文件名
"""
import os
import sys
import traceback

import cv2
import numpy as np


def imread_unicode(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(filepath, img):
    ext = os.path.splitext(filepath)[1].lower() or '.jpg'
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(filepath, 'wb') as f:
            f.write(buf.tobytes())
        return True
    return False


def main():
    if len(sys.argv) < 2:
        print("用法: python test_unicode_path.py <图片路径>")
        sys.exit(1)

    src = sys.argv[1]
    print(f"输入路径: {src}")
    print(f"路径中是否含中文: {any('\u4e00' <= c <= '\u9fff' for c in src)}")
    print(f"文件是否存在: {os.path.isfile(src)}")

    # 步骤1: 读取
    print("\n[1] 尝试读取...")
    img = imread_unicode(src)
    if img is None:
        print("❌ 读取失败！")
        # 试试用原生 cv2.imread 作为对照
        print("    用 cv2.imread 再试一次...")
        img2 = cv2.imread(src)
        if img2 is None:
            print("    cv2.imread 也失败了（这确认了是编码问题）")
        else:
            print(f"    cv2.imread 成功了?! shape={img2.shape}（这说明问题不在路径编码）")
        sys.exit(1)
    print(f"✅ 读取成功! shape={img.shape}, dtype={img.dtype}")

    # 步骤2: 写入
    base = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(os.path.dirname(src) or '.', f"test_output_{base}.png")
    print(f"\n[2] 尝试写入: {dst}")
    ok = imwrite_unicode(dst, img)
    if ok:
        print(f"✅ 写入成功!")
        print(f"    验证文件存在: {os.path.isfile(dst)}")
        # 验证能再读回来
        verify = imread_unicode(dst)
        if verify is not None:
            print(f"    再读验证: shape={verify.shape}, 像素一致={np.allclose(img, verify)}")
    else:
        print("❌ 写入失败！")

    print("\n完成.")


if __name__ == '__main__':
    main()
