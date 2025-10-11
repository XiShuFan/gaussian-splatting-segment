import numpy as np
import os
from PIL import Image
from utils.ply_utils import load_gaussian_ply, save_gaussian_ply
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.neighbors import NearestNeighbors
from utils.sh_color_utils import rgb_to_fdc, fdc_to_rgb

def process_one_pair(args):
    salient_mask_path, pixel_gaussian_path, gs_count = args
    # print(salient_mask_path, pixel_gaussian_path)

    data = np.load(pixel_gaussian_path)
    pixel_gaussian_ids = data["pixel_gaussian_ids"]
    pixel_gaussian_counts = data["pixel_gaussian_counts"]

    img = Image.open(salient_mask_path).convert("RGB")
    width, height = img.size
    # 转换成 numpy 数组（H, W, C）
    img_np = np.array(img)[:, :, :3]

    # reshape 之后保持和图像一样的长宽
    pixel_gaussian_ids = pixel_gaussian_ids.reshape(height, width, -1)
    pixel_gaussian_counts = pixel_gaussian_counts.reshape(height, width)

    # 找到非黑像素的 mask
    # (H, W)，True 表示非黑色像素
    non_black_mask = np.any(img_np >= 230, axis=-1)

    # 使用 non_black_mask 过滤像素
    # (N,)
    valid_counts = pixel_gaussian_counts[non_black_mask]
    # TODO 限定像素对应的高斯数量
    valid_counts[:] = gs_count
    # (N, MAX_GAUSSPERPIXEL)
    valid_ids = pixel_gaussian_ids[non_black_mask, :]

    # 构造 mask，标出哪些 id 位置有效
    mask = np.arange(valid_ids.shape[1])[None, :] < valid_counts[:, None]
    all_valid_ids = valid_ids[mask]

    # 提取所有有效 id，并去重
    return set(all_valid_ids.tolist())


def parallel_collect_ids_mp(salient_mask_folder, pixel_gaussian_folder, gs_count):
    # 排序后的前景分割
    salient_mask_list = sorted(os.listdir(salient_mask_folder))
    # 排序后的高斯
    pixel_gaussian_list = sorted(os.listdir(pixel_gaussian_folder))

    salient_gs_ids = set()
    file_pairs = [
        (os.path.join(salient_mask_folder, sm), os.path.join(pixel_gaussian_folder, pg), gs_count)
        for sm, pg in zip(salient_mask_list, pixel_gaussian_list)
    ]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_one_pair, pair) for pair in file_pairs]

        for f in as_completed(futures):
            salient_gs_ids |= f.result()

    return salient_gs_ids

# 基于欧氏距离的过滤
def filter_by_euclidean_distance(salient_gs_data):
    """
    优点：快速去除远离主体的伪影。
    缺点：如果主体本身很分散（比如多个物体），会损失部分有效点。
    """
    positions = salient_gs_data["position"]
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    # 去掉最远的5%
    threshold = np.percentile(distances, 95)
    mask = distances < threshold
    filtered_salient_gs_data = {k: v[mask] for k, v in salient_gs_data.items()}
    return filtered_salient_gs_data, mask


# 基于密度过滤
def filter_by_density(salient_gs_data):
    """
    优点：可以保留多个主体。
    缺点：运行稍慢。
    """
    positions = salient_gs_data["position"]
    nbrs = NearestNeighbors(n_neighbors=10).fit(positions)
    distances, _ = nbrs.kneighbors(positions)
    mean_d = distances[:, 1:].mean(axis=1)

    # 过滤掉局部稀疏区域的点（密度太低的）
    mask = mean_d < np.percentile(mean_d, 95)
    filtered_salient_gs_data = {k: v[mask] for k, v in salient_gs_data.items()}
    return filtered_salient_gs_data, mask


# 基于前景光栅化次数
def filter_by_mask_consistency():
    """
    多视角 mask（比如从不同角度的前景掩码），可以做一个投影一致性检查：
    对于每个高斯点，看它在多少视角中被投射进前景区域，少于一定比例的就丢掉。
    优点：能精准剔除误匹配点。
    缺点：需要相机参数、多视角 mask。
    """

    # for gaussian in gaussians:
    #     count = sum(is_in_foreground_in_view(gaussian, view) for view in views)
    #     if count / len(views) < 0.3:
    #         drop(gaussian)
    pass


if __name__ == "__main__":
    # 前景高斯索引
    pixel_gaussian_folder = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/3dgs/train/ours_30000/pixel_gaussian"
    salient_mask_folder = "/media/why/新加卷/xsf/U-2-Net/test_data/u2net_results"
    salient_gs_ids = parallel_collect_ids_mp(salient_mask_folder, pixel_gaussian_folder, gs_count=10)
    # 高斯点云
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

    # 基于几何空间的伪影清理
    salient_gs_data, mask = filter_by_euclidean_distance(salient_gs_data)
    index_array = index_array[mask]
    salient_gs_data, mask = filter_by_euclidean_distance(salient_gs_data)
    index_array = index_array[mask]
    salient_gs_data, mask = filter_by_euclidean_distance(salient_gs_data)
    index_array = index_array[mask]
    salient_gs_data, mask = filter_by_euclidean_distance(salient_gs_data)
    index_array = index_array[mask]
    salient_gs_data, mask = filter_by_density(salient_gs_data)
    index_array = index_array[mask]

    # TODO 结合 mask 投影一致性验证

    save_gaussian_ply(salient_gs_data, "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/3dgs/point_cloud/iteration_30000/salient.ply")
    
    # 衣服高斯索引
    cloth_pixel_gaussian_folder = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/part_seg/cloth/pixel_gaussian"
    cloth_mask_folder = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/part_seg/cloth/mask"
    cloth_gs_ids = parallel_collect_ids_mp(cloth_mask_folder, cloth_pixel_gaussian_folder, gs_count=2)
    cloth_index_array = np.array(list(cloth_gs_ids), dtype=np.int64)
    # 与前景高斯做交集
    cloth_index_array = np.intersect1d(cloth_index_array, index_array)
    cloth_gs_data = {
        "position": gs_data["position"][cloth_index_array],
        "normal": gs_data["normal"][cloth_index_array],
        "f_dc": gs_data["f_dc"][cloth_index_array],
        "f_rest": gs_data["f_rest"][cloth_index_array],
        "opacity": gs_data["opacity"][cloth_index_array],
        "scale": gs_data["scale"][cloth_index_array],
        "rotation": gs_data["rotation"][cloth_index_array],
    }
    save_gaussian_ply(cloth_gs_data, "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/3dgs/point_cloud/iteration_30000/cloth.ply")
    
    
    # 衣服换色
    cloth_mask = np.isin(index_array, cloth_index_array)
    mean_sh_color = np.mean(cloth_gs_data["f_dc"], axis=0)
    print(fdc_to_rgb(mean_sh_color))
    new_color = rgb_to_fdc(np.asarray([0, 1, 0]))
    salient_gs_data["f_dc"][cloth_mask] = new_color
    save_gaussian_ply(salient_gs_data, "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/3dgs/point_cloud/iteration_30000/change_color.ply")