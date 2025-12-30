"""
以相机参数渲染高斯模型
"""
import torch
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import PipelineParams
import json
import torchvision
import os
from render_one_camera import get_view_from_camera


def render_cameras(model_path, sh_degree, cam_json_path, pipeline, render_path):
    with torch.no_grad():
        # 初始化高斯模型
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(model_path, [])
    
        # 黑色背景
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        cameras = json.load(open(cam_json_path, "r"))
        
        for cam in cameras:
            # 渲染原始视图
            origin_view = get_view_from_camera(cam)
            render_result = render(origin_view, gaussians, pipeline, background)
            invdepth = render_result["depth"].squeeze(0)
            rendering = render_result["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, cam["img_name"]))
        
    return



if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    args = parser.parse_args()
    
    model_path = "/media/why/新加卷/xsf/商品3DGS/mesh_to_gs/shoes_3dgs.ply"
    cam_json_path = "/media/why/新加卷/xsf/商品3DGS/mesh_to_gs/render_output/cameras.json"
    render_path = "/media/why/新加卷/xsf/商品3DGS/mesh_to_gs/render_output/3dgs_images"
    os.makedirs(render_path, exist_ok=True)
    
    render_cameras(model_path, 0, cam_json_path, pipeline, render_path)
    