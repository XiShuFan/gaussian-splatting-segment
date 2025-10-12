import os
from PIL import Image
import numpy as np

def image_with_mask(image_path, mask_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 获取所有 mask 文件名
    mask_files = [f for f in os.listdir(mask_path)
                  if os.path.isfile(os.path.join(mask_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in mask_files:
        img_file = os.path.join(image_path, fname)
        mask_file = os.path.join(mask_path, fname)

        # 确保原图存在
        if not os.path.exists(img_file):
            print(f"{fname} 在 image_path 中不存在")
            exit(-1)

        # 读取图片与 mask
        image = Image.open(img_file).convert("RGB")
        mask = Image.open(mask_file).convert("L")

        # 转 numpy 数组
        img_np = np.array(image, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)

        # 生成全黑区域的掩码
        black_mask = (mask_np == 0)

        # 在原图中将对应像素变成白色
        img_np[black_mask] = [255, 255, 255]

        # 保存
        out_img = Image.fromarray(img_np)
        out_path = os.path.join(output_path, fname)
        out_img.save(out_path)

        print(f"[完成] 已保存 {out_path}")

if __name__ == '__main__':
    image_path = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/images"
    mask_path = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/masks"
    output_path = "/media/why/新加卷/xsf/商品3DGS/scene/undistorted/image_with_mask"

    image_with_mask(image_path, mask_path, output_path)
