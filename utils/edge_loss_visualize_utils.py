import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# 读图
# =========================
def load_image(path, device='cpu'):
    """
    return: tensor [C, H, W], range [0,1]
    """
    img = Image.open(path).convert("RGB")
    img = T.ToTensor()(img).to(device)
    return img


# =========================
# Sobel
# =========================
def sobel_edge_map(img):
    device = img.device

    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    C, H, W = img.shape
    img = img.view(C, 1, H, W)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)

    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return grad.view(C, H, W)


# =========================
# Laplacian
# =========================
def laplacian_edge_map(img):
    device = img.device

    lap_kernel = torch.tensor(
        [[0,  1,  0],
         [1, -4,  1],
         [0,  1,  0]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    C, H, W = img.shape
    img = img.view(C, 1, H, W)

    lap = F.conv2d(img, lap_kernel, padding=1)
    return lap.view(C, H, W)


# =========================
# 可视化
# =========================
def visualize(img_path):
    img = load_image(img_path)

    sobel = sobel_edge_map(img)
    lap   = laplacian_edge_map(img)

    # 转灰度
    img_gray   = img.mean(dim=0).cpu().numpy()
    sobel_gray = sobel.mean(dim=0).cpu().numpy()
    lap_gray   = lap.abs().mean(dim=0).cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Original (Gray)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sobel_gray, cmap='gray')
    plt.title("Sobel Edge (1st-order)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(lap_gray, cmap='gray')
    plt.title("Laplacian Edge (2nd-order)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# =========================
# main
# =========================
if __name__ == "__main__":
    visualize("test.png")  # ← 换成你的图片路径
