import numpy as np
import os
from PIL import Image
from utils.ply_utils import load_gaussian_ply, save_gaussian_ply
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_one_pair(args):
    salient_mask_path, pixel_gaussian_path = args
    print(salient_mask_path, pixel_gaussian_path)

    data = np.load(pixel_gaussian_path)
    pixel_gaussian_ids = data["pixel_gaussian_ids"]
    pixel_gaussian_counts = data["pixel_gaussian_counts"]

    img = Image.open(salient_mask_path)
    width, height = img.size
    # 转换成 numpy 数组（H, W, C）
    img_np = np.array(img)[:, :, :3]

    # reshape 之后保持和图像一样的长宽
    pixel_gaussian_ids = pixel_gaussian_ids.reshape(height, width, -1)
    pixel_gaussian_counts = pixel_gaussian_counts.reshape(height, width)

    # 找到非黑像素的 mask
    # (H, W)，True 表示非黑色像素
    non_black_mask = np.any(img_np != 0, axis=-1)

    # 使用 non_black_mask 过滤像素
    # (N,)
    valid_counts = pixel_gaussian_counts[non_black_mask]
    # (N, MAX_GAUSSPERPIXEL)
    valid_ids = pixel_gaussian_ids[non_black_mask, :]

    # 构造 mask，标出哪些 id 位置有效
    mask = np.arange(valid_ids.shape[1])[None, :] < valid_counts[:, None]
    all_valid_ids = valid_ids[mask]

    # 提取所有有效 id，并去重
    return set(all_valid_ids.tolist())


def parallel_collect_ids_mp(salient_mask_folder, pixel_gaussian_folder):
    # 排序后的前景分割
    salient_mask_list = sorted(os.listdir(salient_mask_folder))
    # 排序后的高斯
    pixel_gaussian_list = sorted(os.listdir(pixel_gaussian_folder))

    salient_gs_ids = set()
    file_pairs = [
        (os.path.join(salient_mask_folder, sm), os.path.join(pixel_gaussian_folder, pg))
        for sm, pg in zip(salient_mask_list, pixel_gaussian_list)
    ]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_one_pair, pair) for pair in file_pairs]

        for f in as_completed(futures):
            salient_gs_ids |= f.result()

    return salient_gs_ids



if __name__ == "__main__":
    pixel_gaussian_folder = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/3dgs/train/ours_30000/pixel_gaussian"
    salient_mask_folder = "/media/why/新加卷/xsf/U-2-Net/test_data/u2net_results"

    salient_gs_ids = parallel_collect_ids_mp(salient_mask_folder, pixel_gaussian_folder)

    gs_data = load_gaussian_ply("/media/why/新加卷/xsf/商品3DGS/scene/undistorted/3dgs/point_cloud/iteration_30000/point_cloud.ply")

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

    save_gaussian_ply(salient_gs_data, "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/3dgs/point_cloud/iteration_30000/salient.ply")