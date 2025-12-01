import torch


# 获取高斯mask
def get_gaussian_mapping_mask(mask, pixel_gaussian_counts, pixel_gaussian_ids, gaussians):
    # 1. 确保 mask 是 bool tensor，shape 与 height*width 一致
    mask = mask.squeeze(0)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    mask = mask.bool()  # (H, W)

    # 2. image shape
    height, width = mask.shape

    # 3. reshape pixel_gaussian_counts 和 pixel_gaussian_ids
    pixel_gaussian_counts = pixel_gaussian_counts.reshape(height, width)  # (H, W)
    pixel_gaussian_ids = pixel_gaussian_ids.reshape(height, width, -1)   # (H, W, MAX_GAUSSPERPIXEL)

    # 4. 用 mask 取出有效像素
    valid_counts = pixel_gaussian_counts[mask]      # (N,)
    valid_ids = pixel_gaussian_ids[mask, :]         # (N, MAX_GAUSSPERPIXEL)

    # 5. 构造 per-pixel 有效 id mask
    mask_per_pixel = torch.arange(valid_ids.shape[1], device=valid_ids.device)[None, :] < valid_counts[:, None]  # (N, MAX_GAUSSPERPIXEL)

    # 6. 收集所有有效 id
    all_valid_ids = valid_ids[mask_per_pixel]
    all_valid_ids = set(all_valid_ids.tolist())

    N = gaussians._xyz.shape[0]   # 高斯数量

    # 创建布尔 mask (N,)
    train_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
    train_mask[list(all_valid_ids)] = True   # 这些 id 需要训练，其它冻结
    
    return train_mask