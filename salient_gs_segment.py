import numpy as np
import os
from PIL import Image
from utils.ply_utils import load_gaussian_ply, save_gaussian_ply

# 排序后的高斯
pixel_gaussian_folder = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/3dgs/train/ours_30000/pixel_gaussian"
pixel_gaussian_list = sorted(os.listdir(pixel_gaussian_folder))

# 排序后的前景分割
salient_mask_folder = "/media/why/新加卷/xsf/U-2-Net/test_data/u2net_results"
salient_mask_list = sorted(os.listdir(salient_mask_folder))

salient_gs_ids = set()

for salient_mask, pixel_gaussian in zip(salient_mask_list, pixel_gaussian_list):
    data = np.load(os.path.join(pixel_gaussian_folder, pixel_gaussian))
    pixel_gaussian_ids = data["pixel_gaussian_ids"]
    pixel_gaussian_counts = data["pixel_gaussian_counts"]

    img = Image.open(os.path.join(salient_mask_folder, salient_mask))
    width, height = img.size
    # 转换成 numpy 数组（H, W, C）
    img_np = np.array(img)[:, :, :3]

    # reshape 之后保持和图像一样的长宽
    pixel_gaussian_ids = pixel_gaussian_ids.reshape(height, width, -1)
    pixel_gaussian_counts = pixel_gaussian_counts.reshape(height, width)

    # 找到非黑像素的 mask
    non_black_mask = np.any(img_np != 0, axis=-1)  # (H, W)，True 表示非黑色像素
    # 使用 non_black_mask 过滤像素
    valid_counts = pixel_gaussian_counts[non_black_mask]  # (N,)
    valid_ids = pixel_gaussian_ids[non_black_mask, :]  # (N, MAX_GAUSSPERPIXEL)

    # 构造 mask，标出哪些 id 位置有效
    mask = np.arange(valid_ids.shape[1])[None, :] < valid_counts[:, None]

    # 提取所有有效 id，并去重
    all_valid_ids = valid_ids[mask]
    result_set = set(all_valid_ids.tolist())

    salient_gs_ids = set.union(salient_gs_ids, result_set)

gs_data = load_gaussian_ply("")

# 转换为 numpy 数组（整数类型）
index_array = np.array(list(salient_gs_ids), dtype=np.int64)

# 获取前景高斯
salient_gs_data = {
    "position": gs_data["position"][index_array],
    "normal": gs_data["normal"][index_array],
    "f_dc": gs_data["f_dc"][index_array],
    "f_rest": gs_data["f_rest"][index_array],
    "opacity": gs_data["opacity"][index_array],
    "scale": gs_data["scale"][index_array],
    "rotation": gs_data["rotation"][index_array],
}

save_gaussian_ply(salient_gs_data, "")