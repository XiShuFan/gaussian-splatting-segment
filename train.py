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

import os
import torch
import torchvision
import statistics
import math
from random import randint
from utils.loss_utils import l1_loss, ssim, masked_l1_loss
from utils.edge_loss_utils import sobel_edge_loss, laplacian_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.pix_gauss_utils import get_gaussian_mapping_mask
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, is_bbox_locate):
    bbox_iter = opt.iterations
    
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, is_bbox_locate=is_bbox_locate, bbox_iter=bbox_iter)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    
    # 每个视角的损失值
    loss_per_view = {}
    # 每个视角的权重
    loss_weight_per_view = {}

    # 每个视角的逆深度 伪真值
    invdepth_per_view_pseudo_gt = {}
    invdepth_pseudo_gt_factor = 0.7

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            # TODO 回填视角
            additional_views = []
            view_mean_loss = statistics.mean(loss_per_view.values())
            for view_iter in viewpoint_stack:
                view_loss_iter = loss_per_view[view_iter.image_name]
                additional_times = math.ceil(view_loss_iter / view_mean_loss) - 1
                additional_views += [view_iter] * additional_times
            viewpoint_stack += additional_views
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        
        # 当前视角名称
        view_name = viewpoint_cam.image_name

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        foreground_mask = viewpoint_cam.original_mask.cuda()
        if is_bbox_locate:
            zero_mask = foreground_mask.squeeze(0) < 1e-6
            # 掩码外为黑色
            gt_image[:, zero_mask] = 0.0
            gt_mask = None
        else:
            gt_mask = None
        Ll1 = masked_l1_loss(image, gt_image, gt_mask)
        
        if not is_bbox_locate:
            foreground_Ll1 = masked_l1_loss(image, gt_image, foreground_mask)
        
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), gt_mask.unsqueeze(0) if gt_mask is not None else None)
            if not is_bbox_locate:
                foreground_ssim = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), foreground_mask.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image, gt_mask)
            if not is_bbox_locate:
                foreground_ssim = ssim(image, gt_image, foreground_mask)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # 边缘损失
        edge_l1 = sobel_edge_loss(image, gt_image)
        lap_loss = laplacian_loss(image, gt_image)
        loss += 0.9 * edge_l1 + 0.1 * lap_loss
        
        
        if not is_bbox_locate:
            foreground_loss = (1.0 - opt.lambda_dssim) * foreground_Ll1 + opt.lambda_dssim * (1.0 - foreground_ssim)

        # Depth regularization
        Ll1depth_pure = 0.0
        # 如果当前视角误差太大，就不要考虑深度了，首先保证颜色
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable and not (loss_weight_per_view.get(view_name, 1) > 2.0):
            invDepth = render_pkg["depth"]
            if len(invdepth_per_view_pseudo_gt) != 0:
                mono_invdepth = invdepth_per_view_pseudo_gt[view_name]
            else:
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            
            if is_bbox_locate:
                combined_mask = depth_mask
                mono_invdepth[:, zero_mask] = 0.0
            else:
                combined_mask = depth_mask
            
            diff = invDepth  - mono_invdepth
            Ll1depth_pure = torch.abs(diff * combined_mask).sum() / (combined_mask.sum() + 1e-8)
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth * 10.0
            Ll1depth = Ll1depth.item()
            
            if not is_bbox_locate:
                combined_mask = depth_mask * foreground_mask
                diff = invDepth  - mono_invdepth
                foreground_Ll1depth_pure = torch.abs(diff * combined_mask).sum() / (combined_mask.sum() + 1e-8)
                foreground_Ll1depth = depth_l1_weight(iteration) * foreground_Ll1depth_pure 
                foreground_loss += foreground_Ll1depth * 10.0
        else:
            Ll1depth = 0
        
        # 更新当前视角损失
        if not is_bbox_locate:
            loss = loss * 0.6 + foreground_loss * 0.4

        loss_per_view[view_name] = loss_per_view.get(view_name, 0) * 0.6 + loss.item() * 0.4
        # 计算当前所有视角的平均损失，计算权重
        per_view_loss_weight = loss.item() / (sum(loss_per_view.values()) / len(loss_per_view))

        # 指数平滑更新权重
        loss_weight_per_view[view_name] = loss_weight_per_view.get(view_name, 1.0) * 0.6 + per_view_loss_weight * 0.4
        per_view_loss_weight = loss_weight_per_view[view_name]
        
        # 小于1的权重设置为1
        per_view_loss_weight = max(per_view_loss_weight, 1.0)
        loss *= per_view_loss_weight

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

            # 逆深度伪真值
            if iteration > invdepth_pseudo_gt_factor * opt.iterations and len(invdepth_per_view_pseudo_gt) == 0:
                os.makedirs(os.path.join(dataset.model_path, "invdepth_pseudo_gt"), exist_ok=True)
                eval_viewpoint_stack = scene.getTrainCameras().copy()
                for eval_cam in eval_viewpoint_stack:
                    render_pkg = render(eval_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                    invDepth = render_pkg["depth"]
                    invdepth_per_view_pseudo_gt[eval_cam.image_name] = invDepth.detach()
                    torchvision.utils.save_image(invDepth, os.path.join(dataset.model_path, "invdepth_pseudo_gt", eval_cam.image_name))

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                # TODO 只保存mask对应的高斯
                # N = gaussians._xyz.shape[0]
                # valid_eval_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
                # eval_viewpoint_stack = scene.getTrainCameras().copy()
                # for eval_cam in eval_viewpoint_stack:
                #     render_pkg = render(eval_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                #     pixel_gaussian_ids = render_pkg["pixel_gaussian_ids"]
                #     pixel_gaussian_counts = render_pkg["pixel_gaussian_counts"]
                #     gt_mask = viewpoint_cam.original_mask.cuda()
                #     eval_mask = get_gaussian_mapping_mask(gt_mask, pixel_gaussian_counts, pixel_gaussian_ids, gaussians)
                #     valid_eval_mask |= eval_mask
                # valid_eval_gaussians = gaussians.get_sub_gaussian_model(valid_eval_mask)
                
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, is_bbox_locate)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, per_view_loss_weight)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # TODO 允许一个高斯在屏幕上最大投影半径
                    size_threshold = 5 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(max_grad=opt.densify_grad_threshold, 
                                                min_opacity=0.05, extent=scene.cameras_extent, 
                                                max_screen_size=size_threshold, radii=radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

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

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    # 输出每个视角的损失值
    print("loss_pre_view", loss_per_view)
    print("loss_weight_per_view", loss_weight_per_view)
    
    return loss_weight_per_view

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, is_bbox_locate=True)
    training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, is_bbox_locate=False)
    # All done
    print("\nTraining complete.")
    
    os.system(f"python render.py -s {dataset.source_path} --model_path {dataset.model_path} -r 1")
