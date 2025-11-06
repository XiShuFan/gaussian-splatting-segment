import os
import numpy as np
from PIL import Image

def get_logo_gaussian_color(logo_mask_path, pixel_gaussian_path):
    data = np.load(pixel_gaussian_path)
    pixel_gaussian_ids = data["pixel_gaussian_ids"]
    pixel_gaussian_counts = data["pixel_gaussian_counts"]

    img = Image.open(logo_mask_path).convert("RGB")
    width, height = img.size
    # 转换成 numpy 数组（H, W, C）
    img_np = np.array(img)[:, :, :3]

    # reshape 之后保持和图像一样的长宽
    pixel_gaussian_ids = pixel_gaussian_ids.reshape(height, width, -1)
    pixel_gaussian_counts = pixel_gaussian_counts.reshape(height, width)

    # 找到 logo 对应的区域 mask
    logo_mask = np.any((img_np > 0) & (img_np < 255), axis=-1)

    valid_counts = pixel_gaussian_counts[logo_mask]
    MAX_GAUSSPERPIXEL = pixel_gaussian_ids.shape[-1]
    # TODO 限定像素对应的高斯数量
    valid_counts[:] = 10
    # (N, MAX_GAUSSPERPIXEL)
    valid_ids = pixel_gaussian_ids[logo_mask, :]
    valid_colors = img_np[logo_mask]
    valid_colors = np.repeat(valid_colors[:, np.newaxis, :], MAX_GAUSSPERPIXEL, axis=1)

    # 构造 mask，标出哪些 id 位置有效
    mask = np.arange(valid_ids.shape[1])[None, :] < valid_counts[:, None]
    all_valid_ids = valid_ids[mask]
    all_valid_colors = valid_colors[mask]
    
    # 去重，同时保持顺序
    unique_ids, inverse_indices = np.unique(all_valid_ids, return_inverse=True)
    # 向量化求每个唯一 ID 的颜色均值
    unique_colors = np.vstack([
        np.bincount(inverse_indices, weights=all_valid_colors[:, i]) / np.bincount(inverse_indices)
        for i in range(all_valid_colors.shape[1])
    ]).T

    return unique_ids, unique_colors


if __name__ == "__main__":
    logo_mask_path = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/stick_logo/00000.png"
    pixel_gaussian_path = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/stick_logo/00000_pixel_gaussian.npz"
    
    ids, colors = get_logo_gaussian_color(logo_mask_path, pixel_gaussian_path)