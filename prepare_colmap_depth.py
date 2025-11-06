#!/usr/bin/env python
import os
import struct
import numpy as np
import cv2
import json
from tqdm import tqdm

# === 官方 COLMAP 读取函数 ===
def read_array(path):
    """
    读取 COLMAP .bin 深度/法线数组
    返回 shape: (H, W) 或 (H, W, C)
    """
    with open(path, "rb") as fid:
        # 读取文件头 width&height&channels&
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)

    # reshape 并 transpose 得到 (H,W) 或 (H,W,C)
    if channels == 1:
        array = array.reshape((width, height), order="F")
        array = array.T
    else:
        array = array.reshape((width, height, channels), order="F")
        array = np.transpose(array, (1, 0, 2))
    return array.squeeze()


# === 辅助函数：归一化深度值 ===
def normalize_depth(depth):
    """归一化深度到 [0,1] 并返回 scale / offset"""
    valid = depth[depth > 0]
    if len(valid) == 0:
        return depth, 1.0, 0.0
    dmin, dmax = np.percentile(valid, [1, 99])
    scale = 1.0 / (dmax - dmin)
    offset = -dmin * scale
    norm_depth = (depth * scale + offset).clip(0, 1)
    return norm_depth, float(scale), float(offset)


# === 主函数：批量转换 COLMAP depth_maps ===
def convert_colmap_depths(depth_maps_dir, images_dir, output_dir, json_dir):
    """
    depth_maps_dir: COLMAP 输出 depth_maps 文件夹
    images_dir: 对应的原始 RGB 图像文件夹
    output_dir: 输出路径，保存 depth png 和 depth_params.json
    """
    os.makedirs(output_dir, exist_ok=True)
    depth_params = {}

    # 获取原始图像尺寸
    image_sizes = {}
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            H, W = img.shape[:2]
            image_sizes[img_file] = (H, W)

    for file in tqdm(sorted(os.listdir(depth_maps_dir))):
        if not file.endswith(".geometric.bin"):
            continue

        base_name = file.replace(".png.geometric.bin", ".png")
        base_name = base_name.replace(".jpg.geometric.bin", ".jpg")
        if base_name not in image_sizes:
            print(f"⚠️ 找不到对应原始图像: {base_name}, 跳过")
            continue

        H, W = image_sizes[base_name]

        # 使用 COLMAP 官方读取函数
        depth = read_array(os.path.join(depth_maps_dir, file))
        if depth.shape != (H, W):
            print(f"⚠️ 深度图尺寸与原图不匹配: {base_name}, 调整 shape")
            depth = depth.reshape(H, W)

        # 归一化
        norm_depth, scale, offset = normalize_depth(depth)

        # 保存 depth png（16-bit）
        depth_uint16 = (norm_depth * 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(output_dir, base_name), depth_uint16)

        # 保存 scale / offset
        depth_params[base_name] = {"scale": scale, "offset": offset}

    # 写入 depth_params.json
    json_path = os.path.join(json_dir, "depth_params.json")
    with open(json_path, "w") as f:
        json.dump(depth_params, f, indent=4)

    print(f"✅ 完成：{len(depth_params)} 张深度图转换")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 depth_params.json: {json_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--depth_maps_dir', type=str)
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--json_dir', type=str)
    args: Namespace = parser.parse_args()
    convert_colmap_depths(args.depth_maps_dir, args.images_dir, args.output_dir, args.json_dir)

