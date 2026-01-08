import torch
import torch.nn.functional as F

def sobel_edge_loss(pred, gt):
    """
    pred, gt: [C, H, W], value range [0, 1]
    """
    device = pred.device

    # Sobel kernels
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

    def gradient(img):
        C, H, W = img.shape
        img = img.view(C, 1, H, W)

        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)

        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return grad.view(C, H, W)

    pred_grad = gradient(pred)
    gt_grad   = gradient(gt)

    return F.l1_loss(pred_grad, gt_grad)



def laplacian_loss(pred, gt):
    """
    pred, gt: [C, H, W]
    """
    device = pred.device

    lap_kernel = torch.tensor(
        [[0,  1,  0],
         [1, -4,  1],
         [0,  1,  0]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    def laplacian(img):
        C, H, W = img.shape
        img = img.view(C, 1, H, W)
        lap = F.conv2d(img, lap_kernel, padding=1)
        return lap.view(C, H, W)

    pred_lap = laplacian(pred)
    gt_lap   = laplacian(gt)

    return F.l1_loss(pred_lap, gt_lap)
