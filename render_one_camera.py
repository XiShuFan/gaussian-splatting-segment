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

camera = {"width": 518, "height": 291, 
          "position": [0.00011208281757627249, -0.00012430399705829913, 7.592072439092976e-05], 
          "rotation": [
              [0.9999999915140682, 3.127467822933982e-05, -0.0001264664301550828], 
              [-3.127937419038784e-05, 0.999999998821472, -3.7130268041640856e-05], 
              [0.0001264652687688477, 3.713422351736542e-05, 0.9999999913137926]
              ], 
          "fy": 437.1155090332031, "fx": 437.8052062988281}


class ViewpointCamera(NamedTuple):
    FoVx: float
    FoVy: float
    image_height: int
    image_width: int
    world_view_transform: tensor
    full_proj_transform: tensor
    camera_center: tensor
    

def render_one_camera(sh_degree, model_path, save_path, pipeline, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
    with torch.no_grad():
        # 初始化高斯模型
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(model_path, [])

        # 黑色背景
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

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
        
        render_result = render(view, gaussians, pipeline, background)
        rendering = render_result["render"]
        
        os.makedirs(save_path, exist_ok=True)
        
        torchvision.utils.save_image(rendering, os.path.join(save_path, "rendered_image.png"))



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    parser.add_argument("--sh_degree", default=2, type=int)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    render_one_camera(args.sh_degree, args.model_path, args.save_path, pipeline.extract(args))