# 对高斯模型做微调

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, masked_l1_loss
from gaussian_renderer import render, network_gui
import sys
import json
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.pix_gauss_utils import get_gaussian_mapping_mask
from render_one_camera import get_view_from_camera
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, camera_path, train_points_num_path, finetuned_model_path):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    
    # 高斯初始化
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians.load_ply(dataset.model_path, [])
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 准备相机参数
    viewpoint_stack = []
    with open(camera_path, 'r') as f:
        train_cameras = json.load(f)
    for cam in train_cameras:
        image_path = os.path.join(dataset.images, cam["img_name"])
        depth_path = os.path.join(dataset.depths, cam["img_name"].replace('render', 'depth').replace('.png', '.npy'))
        # mask_path = os.path.join(dataset.masks, cam["img_name"])
        viewpoint_stack.append({
            "view": get_view_from_camera(cam),
            "gt_image": transforms.ToTensor()(Image.open(image_path).convert("RGB")),
            "gt_depth": torch.from_numpy(np.load(depth_path)).float(),
            # "gt_mask": torch.from_numpy(np.array((Image.open(mask_path).convert("L")), dtype=np.float32)).unsqueeze(0)[:3, ...] / 255.0
        })
    
    if train_points_num_path is not None and train_points_num_path != "":
        # 准备训练的mask
        with open(train_points_num_path, 'r') as f:
            train_points_num = int(f.read().strip())
            train_mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")
            train_mask[:train_points_num] = True
    else:
        train_mask = torch.ones(gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")
    
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        
        # TODO 不修改颜色
        # gaussians._features_dc.requires_grad_(False)
        # gaussians._features_rest.requires_grad_(False)
        # # 学习率设置为0
        # for group in gaussians.optimizer.param_groups:
        #     if group["name"] in ["f_dc", "f_rest"]:
        #         group["lr"] = 0.0
        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack[rand_idx]

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam["view"], gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        def mask_grad(grad):
            # grad shape = (N, C)
            grad[~train_mask] = 0
            return grad

        gaussians._xyz.register_hook(mask_grad)
        gaussians._scaling.register_hook(mask_grad)
        gaussians._rotation.register_hook(mask_grad)
        gaussians._features_dc.register_hook(mask_grad)
        gaussians._features_rest.register_hook(mask_grad)
        gaussians._opacity.register_hook(mask_grad)

        # Loss
        gt_image = viewpoint_cam["gt_image"].cuda()
        Ll1 = masked_l1_loss(image, gt_image, mask=None)
        
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), None)
        else:
            ssim_value = ssim(image, gt_image, mask=None)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam["gt_depth"].cuda()
            diff = invDepth  - mono_invdepth
            
            # TODO 是否需要mask
            Ll1depth_pure = torch.abs(diff).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            # TODO 不做致密化

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
    
    # 保存微调结果
    gaussians.save_ply(finetuned_model_path)  


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # 需要输入的参数
    parser.add_argument("--camera_path", type=str, required=True, help="训练相机参数路径")
    parser.add_argument("--train_points_num_path", type=str, required=True, help="训练点数量路径")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="微调后模型保存路径")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.camera_path, args.train_points_num_path, args.finetuned_model_path)

    # All done
    print("\nTraining complete.")
