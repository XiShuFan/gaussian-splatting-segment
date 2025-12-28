import sys
import numpy as np
from PIL import Image


def compare_images_rgb(
    img_path_1: str,
    img_path_2: str,
    diff_out_path: str,
    tolerance: int = 0
):
    """
    逐像素比较两张图片的 RGB（忽略 Alpha）

    参数：
        img_path_1, img_path_2 : 输入图片路径
        diff_out_path          : 差异图输出路径
        tolerance              : RGB 允许误差（0 = 严格一致）
    """

    # 读取图片，强制转 RGBA（确保通道一致）
    img1 = Image.open(img_path_1).convert("RGBA")
    img2 = Image.open(img_path_2).convert("RGBA")

    if img1.size != img2.size:
        raise ValueError("两张图片尺寸不一致")

    # 转 numpy
    arr1 = np.asarray(img1, dtype=np.int16)
    arr2 = np.asarray(img2, dtype=np.int16)

    # 只取 RGB，忽略 Alpha
    rgb1 = arr1[..., :3]
    rgb2 = arr2[..., :3]

    # 逐像素 RGB 差异
    diff = np.abs(rgb1 - rgb2)

    if tolerance == 0:
        # 严格模式：只要有一个通道不等就算差异
        diff_mask = np.any(diff != 0, axis=-1)
    else:
        # 容差模式
        diff_mask = np.any(diff > tolerance, axis=-1)

    # 统计信息
    total_pixels = diff_mask.size
    diff_pixels = np.count_nonzero(diff_mask)

    print("======== 对比结果 ========")
    print(f"总像素数      : {total_pixels}")
    print(f"差异像素数    : {diff_pixels}")
    print(f"差异比例 (%)  : {diff_pixels / total_pixels * 100:.4f}")

    if diff_pixels == 0:
        print("✅ RGB 完全一致（在当前容差下）")
    else:
        print("❌ 存在 RGB 差异")

    # 生成差异图
    # 有差异 → 白色，无差异 → 黑色
    diff_img = np.zeros((img1.height, img1.width, 3), dtype=np.uint8)
    diff_img[diff_mask] = [255, 255, 255]

    Image.fromarray(diff_img, mode="RGB").save(diff_out_path)
    print(f"差异图已保存至: {diff_out_path}")


if __name__ == "__main__":
    img1_path = "shoes_3dgs_norm-image.png"
    img2_path = "shoes_3dgs_norm-image-overlay.png"
    diff_path = "diff_img.png"
    tol = 0

    compare_images_rgb(img1_path, img2_path, diff_path, tolerance=tol)
