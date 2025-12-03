#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pix_gauss_utils import get_gaussian_mapping_mask
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import json
import math
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# Example camera info for testing
{"id": 0, "img_name": "frame_00000.png", "width": 518, "height": 291, "position": [0.00011208281757627249, -0.00012430399705829913, 7.592072439092976e-05], "rotation": [[0.9999999915140682, 3.127467822933982e-05, -0.0001264664301550828], [-3.127937419038784e-05, 0.999999998821472, -3.7130268041640856e-05], [0.0001264652687688477, 3.713422351736542e-05, 0.9999999913137926]], "fy": 437.1155090332031, "fx": 437.8052062988281}

import numpy as np
import torch

def pixels_to_world(px, py, Z_cam, view, additional_mask=None):
    # px, py: (...,) pixel coordinates
    # Z_cam:  (...,) invdepth in camera space
    
    valid_mask = Z_cam > 1e-5
    if additional_mask is not None:
        valid_mask = valid_mask & additional_mask
    
    Z_cam = 1.0 / (Z_cam + 1e-8)
    
    W, H = view.image_width, view.image_height

    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)

    cx = W / 2
    cy = H / 2

    # Step 1: normalize to screen coords
    screen_x = (px - cx) / (W / 2)
    screen_y = (py - cy) / (H / 2)

    # Step 2: back-project to camera coordinates
    X_cam = screen_x * Z_cam * tanfovx
    Y_cam = screen_y * Z_cam * tanfovy

    xyz_cam = torch.stack([X_cam, Y_cam, Z_cam], dim=-1)

    # Step 3: camera → world
    R_cw = torch.from_numpy(view.R).float().to(xyz_cam.device)
    C = view.camera_center.to(xyz_cam.device)

    xyz_world = xyz_cam @ R_cw.T + C
    xyz_world = xyz_world[valid_mask]
    return valid_mask, xyz_world



def save_origin_ply(pts, colors, save_path):
    vertex_info = ""
    for point, color in zip(pts, colors):
        vertex_info += f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]} 255\n"
    header = (f"ply\n"
              f"format ascii 1.0\n"
              f"comment VCGLIB generated\n"
              f"element vertex {pts.shape[0]}\n"
              f"property double x\n"
              f"property double y\n"
              f"property double z\n"
              f"property uchar red\n"
              f"property uchar green\n"
              f"property uchar blue\n"
              f"property uchar alpha\n"
              f"end_header\n")
    with open(save_path, 'w', encoding='ascii') as f:
        f.write(header)
        f.write(vertex_info)

    return


def draw_points_red(img, coords):
    """
    img: torch tensor [3, H, W], RGB
    coords: torch tensor [N, 2], (x, y)
    """

    H = img.shape[1]
    W = img.shape[2]

    # 提取 x, y 坐标
    xs = coords[:, 0].long()
    ys = coords[:, 1].long()

    # 去除越界坐标
    mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs = xs[mask]
    ys = ys[mask]

    # 设置红色：R=1, G=0, B=0（若你的图是 0~255，改成 255,0,0）
    img[0, ys, xs] = 1.0
    img[1, ys, xs] = 0.0
    img[2, ys, xs] = 0.0

    return img



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # 像素对应高斯id
    pixel_gaussian_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pixel_gaussian")
    # mask区域对应的第一个高斯点
    mask_gaussian_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask_gaussian")
    # 深度空间像素点
    depth_points_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_points")
    # 相机位置
    camera_path = os.path.join(model_path, name, "ours_{}".format(iteration), "camera")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(pixel_gaussian_path, exist_ok=True)
    makedirs(mask_gaussian_path, exist_ok=True)
    makedirs(depth_points_path, exist_ok=True)
    makedirs(camera_path, exist_ok=True)
    
    cam_list = []
    cam_map = {}
    
    gaussian_xyz = gaussians._xyz.cpu().numpy()
    gaussian_color = np.ones((gaussian_xyz.shape[0], 3)) * 255
    

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_result = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_result["render"]
        gt = view.original_image[0:3, :, :]
        gt_mask = view.original_mask.squeeze(0).bool()
        pixel_gaussian_ids = render_result["pixel_gaussian_ids"]
        pixel_gaussian_counts = render_result["pixel_gaussian_counts"]
        invDepth = render_result["depth"].squeeze(0)
        image_name = view.image_name
        
        # 每个像素渲染的第一个高斯点
        pixel_gaussian_counts[pixel_gaussian_counts > 1] = 1
        
        gaussian_mask = get_gaussian_mapping_mask(view.original_mask, pixel_gaussian_counts, pixel_gaussian_ids, gaussians)
        sub_model = gaussians.get_sub_gaussian_model(gaussian_mask)

        
        ######## 3d高斯到像素 ###########
        _xyz_world = gaussians.get_xyz
        # 筛选出可见的高斯点 (仍是世界坐标)
        # visibility_filter = render_result["visibility_filter"]
        # visible_xyz_world = _xyz_world[visibility_filter.squeeze()]
        visible_xyz_world = _xyz_world

        R_cw = torch.from_numpy(view.R).to(visible_xyz_world.device).float()
        t_cw = view.camera_center

        points_translated_to_camera_origin = visible_xyz_world - t_cw.unsqueeze(0)

        visible_xyz_cam = torch.matmul(points_translated_to_camera_origin, R_cw)

        # 获取可见点的X、Y、Z坐标
        X = visible_xyz_cam[:, 0]  # 所有可见点的X坐标
        Y = visible_xyz_cam[:, 1]  # 所有可见点的Y坐标
        Z = visible_xyz_cam[:, 2]  # 所有可见点的Z坐标

        tanfovx = math.tan(view.FoVx * 0.5)
        tanfovy = math.tan(view.FoVy * 0.5)

        # 使用公式计算屏幕空间坐标
        screen_x = X / (Z * tanfovx)
        screen_y = Y / (Z * tanfovy)

        cx = view.image_width // 2
        cy = view.image_height // 2
        pixel_x = screen_x * (view.image_width / 2) + cx
        pixel_y = screen_y * (view.image_height / 2) + cy
        pixel_points = torch.stack([pixel_x, pixel_y], dim=1) 
        print(pixel_points.shape, rendering.shape)
        rendering_point = draw_points_red(rendering.clone(), pixel_points)
        torchvision.utils.save_image(rendering_point, os.path.join(render_path, image_name))
        
        
        ################# 3D 高斯到像素 ################
        
        torchvision.utils.save_image(gt, os.path.join(gts_path, image_name))
        np.savez_compressed(os.path.join(pixel_gaussian_path, image_name + "_pixel_gaussian.npz"),
                            pixel_gaussian_ids=pixel_gaussian_ids.cpu().numpy(),
                            pixel_gaussian_counts=pixel_gaussian_counts.cpu().numpy())
        
        # 保存子模型
        sub_model.save_ply(os.path.join(mask_gaussian_path, image_name + "_mask_gaussian.ply"))
        print("逆深度值,", invDepth.min().item(), invDepth.max().item(), invDepth.mean().item())
        
        H, W = view.image_height, view.image_width
        # pixel grid
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        ys, xs = ys.to(invDepth.device), xs.to(invDepth.device)
        

        focal_y = H / (2.0 * tanfovy)
        focal_x = W / (2.0 * tanfovx)
        cam_info = {
            "width": W,
            "height": H,
            "fx": focal_x,
            "fy": focal_y,
            "rotation": view.R.tolist(),
            "position": view.camera_center.cpu().numpy().tolist(),
            "world_view_transform": view.world_view_transform.cpu().numpy().tolist(),
            "projection_matrix": view.projection_matrix.cpu().numpy().tolist(),
            "full_proj_transform": view.full_proj_transform.cpu().numpy().tolist()
        }
    
        valid_mask, pts_world = pixels_to_world(xs, ys, invDepth, view, additional_mask=gt_mask)
        
        colors_world = gt[:, valid_mask]
        pts_world = pts_world.cpu().numpy()
        colors_world = colors_world.cpu().numpy().transpose(1, 0) * 255.0
        colors_world = colors_world.astype(np.uint8)
        # 把每个像素的3D点和颜色保存下来
        save_origin_ply(pts_world, colors_world, os.path.join(depth_points_path, image_name + "_depth_points.ply"))
        
        cam_list.append(view.camera_center.cpu().numpy())
        cam_map[image_name] = cam_info
    
    # 保存相机位置
    save_origin_ply(np.asarray(cam_list), np.array([[255, 0, 0]], dtype=np.uint8).repeat(len(cam_list), axis=0), os.path.join(camera_path, "cameras.ply"))
    print(json.dumps(cam_map))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)