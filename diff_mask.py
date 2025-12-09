import cv2
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

def generate_diff_mask(img1_path, img2_path, mask_path):
    # 读取两张图片（BGR）
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError("有图片路径读取失败，请检查")

    # 尺寸不一致时自动 resize
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 计算差异（逐像素判断是否相同）
    diff = cv2.absdiff(img1, img2)

    # 生成 bool 类型 mask（True = 不同）
    mask = np.any(diff != 0, axis=2)   # (H, W) bool

    # 膨胀操作，核为3
    kernel = 3
    structure = np.ones((kernel, kernel), dtype=bool)
    expanded_mask = binary_dilation(mask, structure=structure, iterations=1)

    # 转成 uint8 × 255 保存
    expanded_mask = expanded_mask.astype(np.uint8) * 255

    # 保存 mask
    cv2.imwrite(mask_path, expanded_mask)
    print(f"Mask saved to {mask_path}")


if __name__ == "__main__":
    generate_diff_mask(
        "/media/why/新加卷/xsf/商品3DGS/scene/finetune2/frame_00008_overlay.png",
        "/media/why/新加卷/xsf/商品3DGS/scene/finetune2/frame_00008_origin.png",
        "/media/why/新加卷/xsf/商品3DGS/scene/finetune2/frame_00008_mask.png"
    )
