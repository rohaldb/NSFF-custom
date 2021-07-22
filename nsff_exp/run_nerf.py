import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2

from render_utils import *
from run_nerf_helpers import *
from load_llff import *
from tqdm import tqdm
import torchvision

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def config_parser():
    return dotdict({
        "render_lockcam_slowmo": False,
        "render_slowmo_bt": False,
        "final_height": 288,
        "netdepth": 8,
        "netwidth": 256,
        "netdepth_fine": 8,
        "netwidth_fine": 256,
        "N_rand": 32 * 32 * 4,
        "lrate": 5e-4,
        "lrate_decay": 250,
        "chunk": 1024 * 128,
        "netchunk": 1024 * 128,
        "no_reload": False,
        "ft_path": None,
        "N_samples": 64,
        "N_importance": 0,
        "perturb": 1.,
        "use_viewdirs": False,
        "i_embed": 0,
        "multires": 10,
        "multires_views": 4,
        "raw_noise_std": 0.,
        "render_bt": False,
        "render_test": False,
        "render_factor": 0,
        "render_single_frame": False,
        "skip_blending": False,
        "output_lockcam_flow": False,
        "bt_linear_interpolation": False,
        "dataset_type": 'llff',
        "testskip": 8,
        "white_bkgd": False,
        "factor": 8,
        "no_ndc": False,
        "lindisp": False,
        "spherify": False,
        "llffhold": 8,
        "target_idx": 10,
        "num_extra_sample": 512,
        "decay_depth_w": False,
        "use_motion_mask": False,
        "decay_optical_flow_w": False,
        "w_depth": 0.04,
        "w_optical_flow": 0.02,
        "w_sm": 0.1,
        "w_sf_reg": 0.01,
        "w_cycle": 0.1,
        "w_prob_reg": 0.1,
        "w_sf": 0.1,
        "decay_iteration": 50,
        "start_frame": 0,
        "end_frame": 30,
        "i_print": 500,
        "i_img": 500,
        "i_vid": 10,
        "i_weights": 10000,

        "expname": "kid-running_ndc_5f_sv_of_sm_unify3_testing",

        "basedir": "./logs",
        "datadir": "/content/nerf_data/kid-running/dense",

        "dataset_type": "llff",

        "factor": 2,
        "llffhold": 10,

        "N_rand": 1024,
        "N_samples": 128,
        "N_importance": 0,
        "netwidth": 256,

        "use_viewdirs": True,
        "raw_noise_std": 1e0,
        "no_ndc": False,
        "lindisp": False,
        "no_batching": True,
        "spherify": False,
        "decay_depth_w": True,
        "decay_optical_flow_w": True,
        "use_motion_mask": True,
        "num_extra_sample": 512,
        "decay_iteration": 30,

        "w_depth": 0.04,
        "w_optical_flow": 0.02,
        "w_sm": 0.1,
        "w_sf_reg": 0.01,
        "w_cycle": 1.0,
        "w_prob_reg": 0.1,
        "target_idx": 1,

        "start_frame": 8,
        "end_frame": 11,
    })



def train(config):
    args = config_parser()
    args["render_loss"] = config["render_loss"]
    args["prob_reg_loss"] = config["prob_reg_loss"]
    args["depth_loss"] = config["depth_loss"]
    args["sf_loss"] = config["sf_loss"]
    args["expname"] = args["expname"] + "render:{},prob_reg:{},depth:{},sf:{}".format(config["render_loss"],
                                                                                   config["prob_reg_loss"],
                                                                                   config["depth_loss"],
                                                                                   config["sf_loss"])


    # Load data
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, depths, masks, poses, bds, \
        render_poses, ref_c2w, motion_coords, sf_fw, sf_bw = load_llff_data(args.datadir,
                                                            args.start_frame, args.end_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.final_height)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        i_test = []
        i_val = [] #i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.9 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
    # args.expname = args.expname + '_sigma_rgb-%.2f'%(args.sigma_rgb) \
                # + '_use-rgb-w_' + str(args.use_rgb_w) + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # if args.config is not None:
    #     f = os.path.join(basedir, expname, 'config.txt')
    #     with open(f, 'w') as file:
    #         file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.render_single_frame:
        print("RENDERING SINGLE FRAME")
        num_img = float(poses.shape[0])
        img_idx_embed = target_idx/float(num_img) * 2. - 1.0

        poses = torch.Tensor(poses).to(device)
        chain_bwd = 1
        torch.cuda.empty_cache()
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname,
                                       'rendered_frames_{}_{:06d}'.format('test' if args.render_test else 'path', start))

            render_single_frame(target_idx, img_idx_embed, chain_bwd, num_img,
                                H, W, focal, poses, render_kwargs_train, testsavedir)

        return

    if args.render_bt:
        print('RENDER VIEW INTERPOLATION')
        
        render_poses = torch.Tensor(render_poses).to(device)
        print('target_idx ', target_idx)

        num_img = float(poses.shape[0])
        img_idx_embed = target_idx/float(num_img) * 2. - 1.0

        if args.bt_linear_interpolation:
            render_poses = linearly_interpolate_poses(poses, 10)

        testsavedir = os.path.join(basedir, expname, 
                                'render-spiral-frame-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_bullet_time(render_poses, img_idx_embed, num_img, hwf, 
                               args.chunk, render_kwargs_test, 
                               gt_imgs=images, savedir=testsavedir, 
                               render_factor=args.render_factor)
        return

    if args.render_lockcam_slowmo:
        print('RENDER TIME INTERPOLATION')

        num_img = float(poses.shape[0])
        ref_c2w = torch.Tensor(ref_c2w).to(device)
        pose = poses[target_idx, :3, :4]
        print('target_idx ', target_idx)

        testsavedir = os.path.join(basedir, expname, 'render-lockcam-slowmo')
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_lockcam_slowmo(ref_c2w, num_img, hwf,
                            args.chunk, render_kwargs_train, render_kwargs_test, pose,
                            gt_imgs=images, savedir=testsavedir,
                            render_factor=args.render_factor,
                            target_idx=target_idx,
                            skip_blending=args.skip_blending,
                            output_flow = args.output_lockcam_flow
                            )

            return

    if args.render_slowmo_bt:
        print('RENDER SLOW MOTION')

        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10

        with torch.no_grad():

            testsavedir = os.path.join(basedir, expname, 
                                    'render-slowmo_bt_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            images = torch.Tensor(images)#.to(device)

            print('render poses shape', render_poses.shape)
            render_slowmo_bt(depths, render_poses, bt_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=args.target_idx)

        return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # Move training data to GPU
    images = torch.Tensor(images)#.to(device)
    depths = torch.Tensor(depths)#.to(device)
    sf_fw = torch.tensor(sf_fw)
    sf_bw = torch.tensor(sf_bw)
    poses = torch.Tensor(poses).to(device)

    N_iters = 10000 + 1#500 * 1000 #1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    num_img = float(images.shape[0])
    
    decay_iteration = args.decay_iteration

    chain_bwd = 0


    for i in tqdm(range(start, N_iters)):
        chain_bwd = 1 - chain_bwd
        # Random from one image
        img_i = np.random.choice(i_train)

        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()

        target = images[img_i].cuda()
        pose = poses[img_i, :3,:4]
        depth_gt = depths[img_i].cuda()
        sf_fw_gt = sf_fw[img_i].cuda()
        sf_bw_gt = sf_bw[img_i].cuda()

        hard_coords = torch.Tensor(motion_coords[img_i]).cuda()

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

            if args.use_motion_mask and i < decay_iteration * 1000:
                # print('HARD MINING STAGE !')
                num_extra_sample = args.num_extra_sample
                # print('num_extra_sample ', num_extra_sample)
                select_inds_hard = np.random.choice(hard_coords.shape[0], 
                                                    size=[min(hard_coords.shape[0], 
                                                        num_extra_sample)], 
                                                    replace=False)  # (N_rand,)
                select_inds_all = np.random.choice(coords.shape[0], 
                                                size=[N_rand], 
                                                replace=False)  # (N_rand,)

                select_coords_hard = hard_coords[select_inds_hard].long()
                select_coords_all = coords[select_inds_all].long()

                select_coords = torch.cat([select_coords_all, select_coords_hard], 0)

            else:
                select_inds = np.random.choice(coords.shape[0], 
                                            size=[N_rand], 
                                            replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
            
            rays_o = rays_o[select_coords[:, 0], 
                            select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], 
                            select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_rgb = target[select_coords[:, 0], 
                                select_coords[:, 1]]  # (N_rand, 3)
            target_depth = depth_gt[select_coords[:, 0], 
                                select_coords[:, 1]]
            target_sf_fw = sf_fw_gt[select_coords[:, 0],
                                select_coords[:, 1]]
            target_sf_bw = sf_bw_gt[select_coords[:, 0],
                                    select_coords[:, 1]]

        img_idx_embed = img_i/num_img * 2. - 1.0

        #####  Core optimization loop  #####
        if i < decay_iteration * 1000:
            chain_5frames = False
        else:
            chain_5frames = True

        ret = render(img_idx_embed, chain_bwd, chain_5frames,
                     num_img, H, W, focal, 
                     chunk=args.chunk, rays=batch_rays,
                     verbose=i < 10, retraw=True,
                     **render_kwargs_train)

        optimizer.zero_grad()

        weight_map_post = ret['prob_map_post']
        weight_map_prev = ret['prob_map_prev']

        #disoclusion weight loss
        prob_reg_loss = args.w_prob_reg * (torch.mean(torch.abs(ret['raw_prob_ref2prev'])) \
                                + torch.mean(torch.abs(ret['raw_prob_ref2post'])))

        # dynamic rendering loss
        render_loss = img2mse(ret['rgb_map_ref_dy'], target_rgb)
        render_loss += compute_mse(ret['rgb_map_post_dy'], 
                                   target_rgb, 
                                   weight_map_post.unsqueeze(-1))
        render_loss += compute_mse(ret['rgb_map_prev_dy'], 
                                   target_rgb, 
                                   weight_map_prev.unsqueeze(-1))

        # union rendering loss
        render_loss += img2mse(ret['rgb_map_ref'][:N_rand, ...], 
                            target_rgb[:N_rand, ...])

        divsor = i // (decay_iteration * 1000)

        decay_rate = 10

        if args.decay_depth_w:
            w_depth = args.w_depth/(decay_rate ** divsor)
        else:
            w_depth = args.w_depth

        #depth loss
        depth_loss = w_depth * compute_depth_loss(ret['depth_map_ref_dy'], -target_depth)

        #sf loss
        sf_loss = args.w_sf * compute_sf_loss(target_sf_fw, target_sf_bw,
                                              -ret['sf_map_ref2post'], -ret['sf_map_ref2prev'], ret['weights_ref_dy'],
                                              ret['raw_pts_ref'], H, W, focal)

        if chain_5frames:

            weight_map_pp = ret['prob_map_pp']
            raw_weight_p2pp = (1. - ret['raw_prob_p2pp'])
            prob_reg_loss += args.w_prob_reg * torch.mean(torch.abs(ret['raw_prob_p2pp']))

            render_loss += compute_mse(ret['rgb_map_pp_dy'], 
                                       target_rgb, 
                                       weight_map_pp.unsqueeze(-1))


        loss = render_loss + prob_reg_loss + depth_loss + sf_loss

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

            if args.N_importance > 0:

                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)

            print('Saved checkpoints at', path)


        if i % args.i_print == 0 and i > 0:
            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/render_loss", render_loss.item(), i)
            writer.add_scalar("train/depth_loss", depth_loss.item(), i)
            writer.add_scalar("train/prob_reg_loss", prob_reg_loss.item(), i)
            writer.add_scalar("train/sf_loss", sf_loss.item(), i)

        if i % args.i_img == 0 and i > 0:
            img_i = target_idx
            target = images[img_i]
            pose = poses[img_i, :3, :4]
            target_sf_fw = sf_fw[img_i]
            target_sf_bw = sf_bw[img_i]
            target_depth = depths[img_i] - torch.min(depths[img_i])
            img_idx_embed = img_i/num_img * 2. - 1.0

            with torch.no_grad():
                print("writing images")
                ret = render(img_idx_embed, chain_bwd, False,
                             num_img, H, W, focal,
                             chunk=1024*16, c2w=pose,
                             **render_kwargs_train)

                pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
                pose_prev = poses[max(img_i - 1, 0), :3,:4]
                render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, pose, pose_prev,
                                                                    H, W, focal, ret, n_dim=2)

                render_flow_fwd_rgb = torch.Tensor(flow_to_image(render_of_fwd.cpu().numpy())/255.)
                render_flow_bwd_rgb = torch.Tensor(flow_to_image(render_of_bwd.cpu().numpy())/255.)


                for key in ret.keys():
                    ret[key] = ret[key].to(torch.device("cpu"))

                writer.add_image("val/rgb_map_ref", torch.clamp(ret['rgb_map_ref'], 0., 1.),
                                 global_step=i, dataformats='HWC')
                writer.add_image("val/gt_rgb", target,
                                global_step=i, dataformats='HWC')

                writer.add_image("val/ref_from_prev", torch.clamp(ret['rgb_map_prev_dy'], 0., 1.),
                                 global_step=i, dataformats='HWC')
                writer.add_image("val/ref_from_post", torch.clamp(ret['rgb_map_post_dy'], 0., 1.),
                                 global_step=i, dataformats='HWC')

                writer.add_image("val/gt_sf_fw", compute_color_sceneflow(target_sf_fw),
                                 global_step=i, dataformats='HWC')
                writer.add_image("val/gt_sf_bw", compute_color_sceneflow(target_sf_bw), global_step=i, dataformats='HWC')

                writer.add_image("val/sf_fw_map", compute_color_sceneflow(-ret['sf_map_ref2post']),
                                 global_step=i, dataformats='HWC')
                writer.add_image("val/sf_bw_map", compute_color_sceneflow(-ret['sf_map_ref2prev']),
                                 global_step=i, dataformats='HWC')

                writer.add_image("val/render_opt_flow_fwd_rgb", render_flow_fwd_rgb,
                                 global_step=i, dataformats='HWC')
                writer.add_image("val/render_opt_flow_bwd_rgb", render_flow_bwd_rgb,
                                 global_step=i, dataformats='HWC')

                writer.add_image("val/depth_map_ref", normalize_depth(ret['depth_map_ref']),
                                 global_step=i, dataformats='HW')
                writer.add_image("val/gt_depth_map",
                                    torch.clamp(target_depth /percentile(target_depth, 97), 0., 1.),
                                    global_step=i, dataformats='HW')

        if i%args.i_vid == 0 and i > 0:
            with torch.no_grad():
                # print("writing video")
                # torch.cuda.empty_cache()
                #
                # img_i = target_idx
                # num_img = float(poses.shape[0])
                # ref_c2w = torch.Tensor(ref_c2w).to(device)
                # pose = poses[img_i, :3, :4]
                # img_idx_embed = img_i / num_img * 2. - 1.0
                #
                # # UNCOMMENT TO RENDER TIME INTERPOLATION DURING TRAINING
                #
                # testsavedir = os.path.join(basedir, expname, 'render-lockcam-slowmo')
                # os.makedirs(testsavedir, exist_ok=True)
                # video_path = render_lockcam_slowmo(ref_c2w, num_img, hwf,
                #                                    args.chunk, render_kwargs_train, render_kwargs_test, pose,
                #                                    gt_imgs=images, savedir=testsavedir,
                #                                    render_factor=args.render_factor,
                #                                    target_idx=target_idx,
                #                                    skip_blending=True,
                #                                    output_flow=False
                #                                    )
                #
                # write_video_to_tensorboard(video_path, "lockcam-slomo", i, writer)
                #
                # # uncomment this line to use spiral poses
                # render_poses = torch.Tensor(render_poses).to(device)
                # bt_render_poses = linearly_interpolate_poses(poses.cpu().numpy(), 10)
                # testsavedir = os.path.join(basedir, expname, 'render-spiral-frame')
                # os.makedirs(testsavedir, exist_ok=True)
                # video_path = render_bullet_time(bt_render_poses, img_idx_embed, num_img, hwf,
                #                                 args.chunk, render_kwargs_test,
                #                                 gt_imgs=images, savedir=testsavedir,
                #                                 render_factor=args.render_factor)
                #
                # write_video_to_tensorboard(video_path, "bullet-time", i, writer)

                tune.report(accuracy=compute_accuracy(config))



            torch.cuda.empty_cache()

        global_step += 1

def compute_accuracy(config):
    return config["render_loss"] + config["prob_reg_loss"] + config["depth_loss"] + config["sf_loss"]

#linearly interpolated num_frames poses between poses[1] and poses[-1]. Assumes poses is numpy array
def linearly_interpolate_poses(poses, num_poses):
    fst = poses[0]
    lst = poses[-1]
    render_poses = np.array([((num_poses - t) / (num_poses - 1)) * fst + ((t - 1) / (num_poses - 1)) * lst for t in range(1, num_poses + 1)])
    return torch.Tensor(render_poses).to(device)

def write_video_to_tensorboard(video_path, tag, global_step, writer):
    video = torchvision.io.read_video(video_path)
    v = torch.unsqueeze(video[0], 0)  # T, H, W, C
    # transform to T,C,H,W
    v = torch.swapaxes(v, 2, 4)
    v = torch.swapaxes(v, 3, 4)
    writer.add_video("val/" + tag, vid_tensor=v, global_step=global_step, fps=20)

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    config = {
        "sf_loss": tune.choice([2, 4, 8, 16]),
        "prob_reg_loss": tune.choice([0.1]),
        "depth_loss": tune.choice([0.04, 0.1, 0.2, 0.4]),
        "render_loss": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"])
    # result = tune.run(
    #     train,
    #     resources_per_trial={"gpu": gpus_per_trial},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    #     progress_reporter=reporter)

    result = tune.run(train, config=config, resources_per_trial={"cpu": 0, "gpu": 1})

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main(num_samples=1, max_num_epochs=10, gpus_per_trial=1)
