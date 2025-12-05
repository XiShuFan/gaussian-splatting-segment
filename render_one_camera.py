"""
以一个随机的相机位姿，渲染高斯模型的图片
"""
from argparse import ArgumentParser
from arguments import PipelineParams, get_combined_args
import torch
from torch import tensor
import numpy as np
import torchvision
import os
from typing import NamedTuple
from numpy import ndarray
from PIL import Image
import torch.nn.functional as F

from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import focal2fov, getWorld2View2, compute_t_from_R_and_center, getProjectionMatrix
from utils.pix_gauss_utils import get_gaussian_mapping_mask
from utils.general_utils import PILtoTorch, load_mask_as_tensor
from transfer_logo_to_other_view import pixels_to_world, save_origin_ply
from scene.dataset_readers import fetchPly


class ViewpointCamera(NamedTuple):
    FoVx: float
    FoVy: float
    image_height: int
    image_width: int
    world_view_transform: tensor
    full_proj_transform: tensor
    camera_center: tensor
    R: ndarray
    

def normalize(v):
    return v / np.linalg.norm(v)

def look_at(eye, target, up):
    forward = normalize(target - eye)
    right = normalize(np.cross(up, forward))
    up_new = np.cross(forward, right)
    R = np.stack([right, up_new, forward], axis=1)
    return R

def sample_cameras_on_circle(camera, target_center, radius=0.1, n=8):
    cameras = []

    # 原始相机参数
    R = np.array(camera["rotation"])
    C = np.array(camera["position"])
    up = R[:, 1]     # 上方向
    target = np.array(target_center)

    # 法向量：相机到物体
    forward = normalize(target - C)

    # 在法平面上构建两个正交基底 u、v
    u = normalize(np.cross(up, forward))
    v = np.cross(forward, u)

    # 均匀采样 n 个角度
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    for theta in angles:
        # 圆周上点的偏移
        offset = radius * (np.cos(theta) * u + np.sin(theta) * v)

        # 新相机中心
        new_C = C + offset

        # 新相机朝向：仍然看向物体中心
        new_R = look_at(new_C, target, up)

        # 组装新的 camera
        new_camera = camera.copy()
        new_camera["position"] = new_C
        new_camera["rotation"] = new_R

        cameras.append(new_camera)

    return cameras



def get_view_from_camera(camera, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
    FoVy = focal2fov(camera["fy"], camera["height"])
    FoVx = focal2fov(camera["fx"], camera["width"])
    
    R = np.array(camera["rotation"])
    camera_center = np.array(camera["position"])
    T = compute_t_from_R_and_center(R, camera_center)
    
    # TODO 计算投影矩阵
    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    
    view: ViewpointCamera = ViewpointCamera(
        FoVx=FoVx,
        FoVy=FoVy,
        image_width=camera["width"],
        image_height=camera["height"],
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
        R=R
    )
    return view


# 一个像素想保留为 1，邻域内必须全部为 1；否则就变成 0
def erode_mask(mask, k):
    """
    mask: (H, W) boolean mask
    k: 腐蚀半径（像素）
    """
    if k <= 0:
        return mask
    
    # convert to float so MinPool2d can process
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # kernel size = 2*k+1 (例如收缩 3 像素 -> kernel = 7)
    kernel = 2 * k + 1

    # MinPool2d = erosion
    eroded = -F.max_pool2d(-mask_f, kernel_size=kernel, stride=1, padding=k)

    eroded_mask = (eroded[0,0] > 0.999)  # convert back to bool
    return eroded_mask



def render_one_camera(sh_degree, model_path, save_path, pipeline, camera, image_logo_path, mask_logo_path, logo_ply_path, logo_gaussian_path, full_gaussian_path):
    with torch.no_grad():
        # 初始化高斯模型
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(model_path, [])
        os.makedirs(save_path, exist_ok=True)
        
        # 读取 logo 图像和 mask
        W, H = camera["width"], camera["height"]
        logo_image = PILtoTorch(Image.open(image_logo_path), (W, H))
        gt_mask = load_mask_as_tensor(Image.open(mask_logo_path).convert('L'), (W, H))

        # 黑色背景
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 渲染原始视图
        origin_view = get_view_from_camera(camera)
        render_result = render(origin_view, gaussians, pipeline, background)
        invdepth = render_result["depth"].squeeze(0)
        rendering = render_result["render"]
        torchvision.utils.save_image(rendering, os.path.join(save_path, "origin_image.png"))
        
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        ys = ys.to(invdepth.device)
        xs = xs.to(invdepth.device)
        gt_mask = gt_mask.bool().squeeze(0).to(invdepth.device)
        valid_mask, pts_world = pixels_to_world(xs, ys, invdepth, origin_view, additional_mask=gt_mask)
        colors_world = logo_image[:, valid_mask]
        pts_world = pts_world.cpu().numpy()
        colors_world = colors_world.cpu().numpy().transpose(1, 0) * 255.0
        colors_world = colors_world.astype(np.uint8)
        # 保存logo点云
        save_origin_ply(pts_world, colors_world, logo_ply_path)
        
        # TODO 获取视线中心（可以考虑使用mask的中心点坐标）
        pixel_gaussian_ids = render_result["pixel_gaussian_ids"]
        pixel_gaussian_counts = render_result["pixel_gaussian_counts"]
        pixel_gaussian_counts[pixel_gaussian_counts > 1] = 1
        mask = torch.zeros((H, W), dtype=torch.bool)
        center_h, center_w = H // 2, W // 2
        mask[center_h-5:center_h+5, center_w-5:center_w+5] = 1
        gaussian_mask = get_gaussian_mapping_mask(mask, pixel_gaussian_counts, pixel_gaussian_ids, gaussians)
        sub_model = gaussians.get_sub_gaussian_model(gaussian_mask)
        target_center = sub_model.get_xyz.mean(dim=0).cpu().numpy()
        # 获取结束
        
        # 从logo点云初始化高斯模型
        logo_gaussians = GaussianModel(sh_degree)
        logo_gaussians.create_from_pcd(fetchPly(logo_ply_path), [], 1, full_opacity=True)
        
        logo_render_result = render(origin_view, logo_gaussians, pipeline, background)
        logo_rendering = logo_render_result["render"]
        torchvision.utils.save_image(logo_rendering, os.path.join(save_path, "origin_logo.png"))
        
        # 渲染绕圈视图
        dist = np.linalg.norm(np.array(camera["position"]) - target_center)
        radius = dist * 0.2
        new_camera_list = sample_cameras_on_circle(camera, target_center, radius=radius, n=10)
        for i, new_camera in enumerate(new_camera_list):
            # 新相机位姿
            new_view = get_view_from_camera(new_camera)
            # 渲染logo
            logo_render_result = render(new_view, logo_gaussians, pipeline, background)
            logo_rendering = logo_render_result["render"]
            # 渲染原场景
            render_result = render(new_view, gaussians, pipeline, background)
            rendering = render_result["render"]
            # logo叠加到原场景
            mask = (logo_rendering.abs().sum(dim=0) > 0)
            kernel = 3
            mask_eroded = erode_mask(mask, kernel)
            mask3 = mask_eroded.unsqueeze(0).expand_as(rendering)
            rendering[mask3] = logo_rendering[mask3]
            # 保存图片
            torchvision.utils.save_image(rendering, os.path.join(save_path, f"perturbed_render_{i:02d}.png"))
            
        # 保存logo高斯场景
        logo_gaussians.save_ply(logo_gaussian_path)
        # 合并模型并保存
        merge_model = gaussians.merge_gaussian_model(logo_gaussians)
        merge_model.save_ply(full_gaussian_path)
    return


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    parser.add_argument("--sh_degree", default=2, type=int, help="球谐阶数")
    parser.add_argument("--model_path", type=str, required=True, help="高斯模型路径")
    parser.add_argument("--save_path", type=str, required=True, help="渲染图片保存路径")
    parser.add_argument("--image_logo_path", type=str, required=True, help="包含logo的图片路径")
    parser.add_argument("--mask_logo_path", type=str, required=True, help="logo的mask路径")
    parser.add_argument("--logo_ply_path", type=str, required=True, help="logo点云保存路径")
    parser.add_argument("--logo_gaussian_path", type=str, required=True, help="logo高斯场景保存路径")
    parser.add_argument("--full_gaussian_path", type=str, required=True, help="合并后高斯场景保存路径")
    args = parser.parse_args()
    print("Rendering " + args.model_path)
    
    camera = {
        "id": 8, 
        "img_name": "frame_00008.png", 
        "width": 518, 
        "height": 291, 
        "position": [0.749573911580539, -0.21718518084729227, 0.8828316220418042], 
        "rotation": [
            [-0.19922023739973, 0.2614259939469231, -0.9444405000939621], 
            [-0.08358388577128692, 0.9557141004165395, 0.2821777745015246], 
            [0.9763837066278003, 0.1351555106983216, -0.1685465930408849]
            ], 
        "fy": 451.1372375488282, 
        "fx": 451.02505493164057
        }

    render_one_camera(args.sh_degree, args.model_path, args.save_path, pipeline.extract(args), 
                      camera, args.image_logo_path, args.mask_logo_path, args.logo_ply_path, args.logo_gaussian_path, args.full_gaussian_path)