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

from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import focal2fov, getWorld2View2, compute_t_from_R_and_center, getProjectionMatrix
from utils.pix_gauss_utils import get_gaussian_mapping_mask


class ViewpointCamera(NamedTuple):
    FoVx: float
    FoVy: float
    image_height: int
    image_width: int
    world_view_transform: tensor
    full_proj_transform: tensor
    camera_center: tensor
    

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
        camera_center=camera_center
    )
    return view


def render_one_camera(sh_degree, model_path, save_path, pipeline, camera):
    with torch.no_grad():
        # 初始化高斯模型
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(model_path, [])
        os.makedirs(save_path, exist_ok=True)

        # 黑色背景
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 渲染原始视图
        origin_view = get_view_from_camera(camera)
        render_result = render(origin_view, gaussians, pipeline, background)
        rendering = render_result["render"]
        
        # TODO 获取视线中心（可以考虑使用mask的中心点坐标）
        pixel_gaussian_ids = render_result["pixel_gaussian_ids"]
        pixel_gaussian_counts = render_result["pixel_gaussian_counts"]
        pixel_gaussian_counts[pixel_gaussian_counts > 1] = 1
        H, W = rendering.shape[1], rendering.shape[2]
        mask = torch.zeros((H, W), dtype=torch.bool)
        center_h, center_w = H // 2, W // 2
        mask[center_h-5:center_h+5, center_w-5:center_w+5] = 1
        gaussian_mask = get_gaussian_mapping_mask(mask, pixel_gaussian_counts, pixel_gaussian_ids, gaussians)
        sub_model = gaussians.get_sub_gaussian_model(gaussian_mask)
        target_center = sub_model.get_xyz.mean(dim=0).cpu().numpy()
        # 获取结束
        
        torchvision.utils.save_image(rendering, os.path.join(save_path, "origin_render.png"))
        
        # 渲染绕圈视图
        dist = np.linalg.norm(np.array(camera["position"]) - target_center)
        radius = dist * 0.1
        new_camera_list = sample_cameras_on_circle(camera, target_center, radius=radius, n=10)
        for i, new_camera in enumerate(new_camera_list):
            new_view = get_view_from_camera(new_camera)
            render_result = render(new_view, gaussians, pipeline, background)
            rendering = render_result["render"]
            torchvision.utils.save_image(rendering, os.path.join(save_path, f"perturbed_render_{i:02d}.png"))
    return


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    parser.add_argument("--sh_degree", default=2, type=int)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    print("Rendering " + args.model_path)
    
    camera = {"width": 518, "height": 291, 
          "position": [0.00011208281757627249, -0.00012430399705829913, 7.592072439092976e-05], 
          "rotation": [
              [0.9999999915140682, 3.127467822933982e-05, -0.0001264664301550828], 
              [-3.127937419038784e-05, 0.999999998821472, -3.7130268041640856e-05], 
              [0.0001264652687688477, 3.713422351736542e-05, 0.9999999913137926]
              ], 
          "fy": 437.1155090332031, "fx": 437.8052062988281}

    render_one_camera(args.sh_degree, args.model_path, args.save_path, pipeline.extract(args), camera)