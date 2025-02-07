import os, sys, copy, glob, json, time, random, argparse, cv2
from shutil import copyfile
from tqdm import tqdm, trange
import math
import mmcv
import imageio
import numpy as np
import trimesh
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from lib import utils, dtu_eval
from torch.utils.tensorboard import SummaryWriter
from lib.load_data import load_data
from lib.utils import rgb_to_luminance, get_sobel, calc_grad, \
    GradLoss, write_ply, load_point_cloud, get_root_logger
from lib.fields import BendingNetwork
from torch_efficient_distloss import flatten_eff_distloss
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

torch.autograd.set_detect_anomaly(True)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--export_fine_only", type=str, default='')
    parser.add_argument("--mesh_from_sdf", action='store_true')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", default=True)
    parser.add_argument("--eval_lpips_alex", default=True)
    parser.add_argument("--eval_lpips_vgg", default=True)

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_validate", type=int, default=1000) #10000)
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("-s", "--suffix", type=str, default="",
                        help='suffix for exp name')
    parser.add_argument("-p", "--prefix", type=str, default="",
                        help='prefix for exp name')
    parser.add_argument("--load_density_only", type=int, default=1)
    parser.add_argument("--load_expname", type=str, default="") # dvgo_Statues_original
    parser.add_argument("--sdf_mode", type=str, default="density")
    parser.add_argument("--scene", type=str, default=0)
    parser.add_argument("--no_dvgo_init", action='store_true')
    parser.add_argument("--run_dvgo_init", action='store_true')
    parser.add_argument("--interpolate", default='')
    parser.add_argument("--extract_color", action='store_true')
    return parser


@torch.no_grad()
@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, masks=None, savedir=None, render_factor=0, idx=None,
                      eval_ssim=True, eval_lpips_alex=True, eval_lpips_vgg=True,
                      use_bar=True, step=0, rgb_only=False, frame=-1):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''

    # render_poses['far'] = 2.7395265, 是正常的
    # __import__('ipdb').set_trace()

    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    normals = []
    ins = []
    outs = []
    disps = []
    psnrs = []
    fore_psnrs = []
    bg_psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    render_normal = True
    split_bg = getattr(model, "bg_density", False)
    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]

        # 到这里两个版本的rays_o和rays_d以及viewdirs还是相同的
        rays_o, rays_d, viewdirs = Model.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'disp', 'alphainv_cum']
        if render_normal:
            keys.append('normal_marched')
        if split_bg:
            keys.extend(['in_marched', 'out_marched'])
        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)

        # __import__('ipdb').set_trace()
        '''
        model -> Voxurf
        IndexError: index 0 is out of bounds for dimension 1 with size 0

        ipdb> keys
        ['rgb_marched', 'disp', 'alphainv_cum', 'normal_marched']
        '''
        # 这里好怪啊，单次跟几轮循环都是可以的，但是就是没法运行完
        # render_result_chunks = [
        #     {k: v for k, v in model(ro, rd, vd, frame, is_train=False, **render_kwargs).items() if k in keys}
        #     for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        # ]

        # 为了方便debug我们先换一种写法
        
        # __import__('ipdb').set_trace()
        
        j = 0
        render_result_chunks = []
        for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0)):
            
            print("#----------i:{}-----------#".format(j))
            # if i == 45:
                # __import__('ipdb').set_trace()
            
            res = model(ro, rd, vd, frame, is_train=False, **render_kwargs)
            render_result_chunks.append(
                {k: v for k, v in res.items() if k in keys}
            )

            j = j + 1
        
        # __import__('ipdb').set_trace()
        
        # __import__('ipdb').set_trace()

        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        rgbs.append(rgb)
        if rgb_only and savedir is not None:
            imageio.imwrite(os.path.join(savedir, '{:03d}.png'.format(i)), utils.to8b(rgb))
            continue

        disp = render_result['disp'].cpu().numpy()
        disps.append(disp)

        if render_normal:
            normal = render_result['normal_marched'].cpu().numpy()
            normals.append(normal)

        if split_bg:
            inside = render_result['in_marched'].cpu().numpy()
            ins.append(inside)
            outside = render_result['out_marched'].cpu().numpy()
            outs.append(outside)

        if masks is not None:
            if isinstance(masks[i], torch.Tensor):
                mask = masks[i].cpu().numpy() #.reshape(H, W, 1)
            else:
                mask = masks[i] #.reshape(H, W, 1)
            if mask.ndim == 2:
                mask = mask.reshape(H, W, 1)
            bg_rgb = rgb * (1 - mask)
            bg_gt = gt_imgs[i] * (1 - mask)
        else:
            mask, bg_rgb, bg_gt = np.ones(rgb.shape[:2]), np.ones(rgb.shape), np.ones(rgb.shape)

        if i==0:
            logger.info('Testing {} {}'.format(rgb.shape, disp.shape))
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            back_p, fore_p = 0., 0.
            if  masks is not None:
                back_p = -10. * np.log10(np.sum(np.square(bg_rgb - bg_gt))/np.sum(1-mask))
                fore_p = -10. * np.log10(np.sum(np.square(rgb - gt_imgs[i]))/np.sum(mask))
            error = 1 - np.exp(-20 * np.square(rgb - gt_imgs[i]).sum(-1))[...,None].repeat(3,-1)

            print("{} | full-image psnr {:.2f} | foreground psnr {:.2f} | background psnr: {:.2f} ".format(i, p, fore_p, back_p))
            psnrs.append(p)
            fore_psnrs.append(fore_p)
            bg_psnrs.append(back_p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device='cpu'))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device='cpu'))
        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            id = idx if idx is not None else i
            step_pre = str(step) + '_' if step > 0 else ''
            filename = os.path.join(savedir, step_pre+'{:03d}.png'.format(id))
            rendername = os.path.join(savedir, step_pre + 'render_{:03d}.png'.format(id))
            gtname = os.path.join(savedir, step_pre + 'gt_{:03d}.png'.format(id))

            img8 = rgb8
            if gt_imgs is not None:
                error8 = utils.to8b(error)
                gt8 = utils.to8b(gt_imgs[i])
                imageio.imwrite(gtname, gt8)
                img8 = np.concatenate([error8, rgb8, gt8], axis=0)

            if split_bg and gt_imgs is not None:
                in8 = utils.to8b(ins[-1])
                out8 = utils.to8b(outs[-1])
                img8_2 = np.concatenate([in8, out8], axis=1)
                img8 = np.concatenate([rgb8, gt8], axis=1)
                img8 = np.concatenate([img8, img8_2], axis=0)

            imageio.imwrite(rendername, rgb8)
            imageio.imwrite(filename, img8)

            if render_normal:
                rot = c2w[:3, :3].permute(1, 0).cpu().numpy()
                normal = (rot @ normals[-1][..., None])[...,0]
                normal = 0.5 - 0.5 * normal
                if masks is not None:
                    normal = normal * mask.mean(-1)[...,None] + (1 - mask)
                normal8 = utils.to8b(normal)
                step_pre = str(step) + '_' if step > 0 else ''
                filename = os.path.join(savedir, step_pre+'{:03d}_normal.png'.format(id))
                imageio.imwrite(filename, normal8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)
    if len(psnrs):
        logger.info('Testing psnr {:.2f} (avg) | foreground {:.2f} | background {:.2f}'.format(
            np.mean(psnrs), np.mean(fore_psnrs), np.mean(bg_psnrs)))
        if eval_ssim: logger.info('Testing ssim {} (avg)'.format(np.mean(ssims)))
        if eval_lpips_vgg: logger.info('Testing lpips (vgg) {} (avg)'.format(np.mean(lpips_vgg)))
        if eval_lpips_alex: logger.info('Testing lpips (alex) {} (avg)'.format(np.mean(lpips_alex)))

    return rgbs, disps


def gen_poses_between(pose_0, pose_1, ratio):
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    return pose


def interpolate_view(savedir, img_idx_0, img_idx_1, render_poses, HW, Ks, ndc, repeat=1, **render_kwargs):
    render_poses = render_poses.cpu().numpy()
    pose_0, pose_1 = render_poses[img_idx_0], render_poses[img_idx_1]
    images = []
    n_frames = 60
    image_dir = os.path.join(savedir, 'images_full')
    os.makedirs(image_dir, exist_ok=True)
    poses = []
    for i in range(n_frames):
        new_pose = gen_poses_between(pose_0, pose_1, np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5)
        poses.append(new_pose)

    render_kwargs.update(dict(
        savedir=image_dir,
        eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
        rgb_only=True,
    ))
    HW = HW[:1].repeat(len(poses),0)
    Ks = Ks[:1].repeat(len(poses),0)
    rgbs, _ = render_viewpoints(render_poses=torch.from_numpy(np.asarray(poses)).cuda(),
                                HW=HW, Ks=Ks, ndc=ndc, **render_kwargs)
    for i in range(n_frames):
        images.append(rgbs[i])
    for i in range(n_frames):
        images.append(rgbs[n_frames - i - 1])
    h, w, _ = images[0].shape
    imageio.mimwrite(os.path.join(savedir, 'render_{}_{}.mp4'.format(img_idx_0, img_idx_1)),
                     utils.to8b(images), fps=30, quality=8)


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    mode = getattr(cfg.data, 'mode', dict())
    if 'train_all' in cfg:
        mode.update(train_all=cfg.train_all)
        print(" * * * Train with all the images: {} * * * ".format(cfg.train_all))
    if 'reso_level' in cfg:
        mode.update(reso_level=cfg.reso_level)
    
    # 调用load_data
    data_dict = load_data(cfg.data, **mode, white_bg=cfg.data.white_bkgd)

    # remove useless field
    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far',
        'i_train', 'i_val', 'i_test', 'irregular_shape',
        'poses', 'render_poses', 'images', 'scale_mats_np', 'masks'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu').cuda() for im in data_dict['images']]
        data_dict['masks'] = [torch.FloatTensor(im, device='cpu').cuda() for im in data_dict['masks']]
    else:
        # custom的数据会走这一条分支
        # 把数据加载到cuda
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu').cuda()
        data_dict['masks'] = torch.FloatTensor(data_dict['masks'], device='cpu').cuda()
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    logger.info('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        # 得到rays_o，rays_d以及viewdirs
        # 该函数来自于voxurf_coarse.py
        rays_o, rays_d, viewdirs = Model.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    logger.info('compute_bbox_by_cam_frustrm: xyz_min {}'.format(xyz_min))
    logger.info('compute_bbox_by_cam_frustrm: xyz_max {}'.format(xyz_max))
    logger.info('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    # 这里还是用的coarse_last.tar算的，不是surf_last.tar，不是特别懂
    logger.info('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()

    # __import__('ipdb').set_trace()

    model = utils.load_model(model_class, model_path, strict=False)
    '''
    interp [161, 76, 82, 3]
    model.density [1, 1, 161, 76, 82]
    '''
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    # 采样点的坐标 [161, 76, 82, 3]

    # __import__('ipdb').set_trace()

    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    # 采样点对应的density [161, 76, 82]
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    logger.info('compute_bbox_by_coarse_geo: xyz_min {}'.format(xyz_min))
    logger.info('compute_bbox_by_coarse_geo: xyz_max {}'.format(xyz_max))
    eps_time = time.time() - eps_time
    logger.info('compute_bbox_by_coarse_geo: finish (eps time: {} secs)'.format(eps_time))
    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None, use_dvgo=False):
    logger.info("= "*10 + "Begin training state [ {} ]".format(stage) + " ="*10)
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        # coarse在DVGO init时没有进这个分支, 之后coarse和fine都进了
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    # 从之前处理的数据字典中得到需要的数据
    # 到这里far也是正常的
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, masks = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'masks'
        ]
    ]
    # coarse阶段在DVGO init的时候是用所有的图片来训练
    print("Train idx", i_train, "\nTest idx", i_test)

    # __import__('ipdb').set_trace()

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        # 不reload检查点
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif getattr(cfg_train, 'ft_path', ''):
        reload_ckpt_path = cfg_train.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model
    # 在这里初始化网络，那我们也先加到这里吧
    model_kwargs = copy.deepcopy(cfg_model)
    scale_ratio = getattr(cfg_train, 'scale_ratio', 2)
    num_voxels = model_kwargs.pop('num_voxels')
    num_voxels_bg = model_kwargs.pop('num_voxels_bg', num_voxels)
    if len(cfg_train.pg_scale) and not args.render_only:
        deduce = (scale_ratio**len(cfg_train.pg_scale))
        num_voxels = int(num_voxels / deduce)
        num_voxels_bg = int(num_voxels_bg / deduce)
        logger.info("\n" + "+ "*10 + "start with {} resolution deduction".format(deduce) + " +"*10 + "\n")
    else:
        deduce = 1

    # __import__('ipdb').set_trace()

    # 计算图片数目
    n_images = len(data_dict['images'])
    # 加到参数里
    model_kwargs['n_images'] = n_images

    if use_dvgo:
        # 带mask的在coarse的init阶段会进入这个分支, model是DirectVoxGO()
        # use dvgo init for the w/ mask setting
        model = dvgo_ori.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            num_voxels_bg=num_voxels_bg,
            mask_cache_path=coarse_ckpt_path,
            exppath=os.path.join(cfg.basedir, cfg.expname),
            **model_kwargs)
    else:
        # 后续的coarse阶段和fine阶段都会进入这个分支
        # coarse-Model: lib.voxurf_coarse
        # fine-Model: lib.voxurf_fine
        model = Model.Voxurf(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            num_voxels_bg=num_voxels_bg,
            mask_cache_path=coarse_ckpt_path,
            exppath=os.path.join(cfg.basedir, cfg.expname),
            **model_kwargs)
    if cfg_model.maskout_near_cam_vox:
        model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    model = model.to(device)

    # __import__('ipdb').set_trace()

    # model = model + model.bending_latents_list
    # init optimizer，定义优化器的时候这些参数就都传进去了
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0, bending_latents_list=model.bending_latents_list)

    load_density_from = getattr(cfg_train, 'load_density_from', '')
    load_sdf_from = getattr(cfg_train, 'load_sdf_from', '')
    if load_density_from and stage == 'surf':
        density_ckpt_path = os.path.join(cfg.basedir, load_density_from)
        if args.load_density_only:
            model = utils.load_grid_data(model, density_ckpt_path, deduce)
        else:
            reload_ckpt_path = density_ckpt_path

    if reload_ckpt_path is None:
        # coarse会进这个分支
        logger.info(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
    else:
        logger.info(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer, strict=False)
        logger.info("Restart from iteration {}, model sdf size: {}".format(start, model.sdf.grid.shape))
        if reload_ckpt_path.split('/')[-1].split('_')[0] != stage:
            start = 0

    if cfg_train.get('load_param', False):
        model, _, _ = utils.load_checkpoint(
            model, None, cfg_train.load_sdf_from, True, strict=False)

    # init sdf
    if load_sdf_from:
        # coarse一直没有进这个分支，fine进了
        if hasattr(model, 'init_sdf_from_sdf'):
            sdf_reduce = cfg_train.get('sdf_reduce', 1.0)
            if cfg_train.load_sdf_from == 'auto':
                cfg_train.load_sdf_from = os.path.join(cfg.basedir, cfg.expname0, 'coarse', 'surf_last.tar')
            if cfg_train.get('load_sdf_path', None):
                cfg_train.load_sdf_from = cfg_train.load_sdf_path + 'scan_{}/surf_last.tar'.format(args.scene)
            logger.info("\n" + "+ "*10 + "load sdf from: " + cfg_train.load_sdf_from + "+"*10 + "\n")
            sdf0 = utils.load_grid_data(model, cfg_train.load_sdf_from, name='sdf', return_raw=True)
            model.init_sdf_from_sdf(sdf0, smooth=False, reduce=sdf_reduce)
            if cfg_train.get('load_bg_all', False):
                bg_density0 = utils.load_grid_data(model, cfg_train.load_sdf_from, name='bg_density', return_raw=True)
                model.init_bg_density_from_bg_density(bg_density0)
                utils.load_weight_by_name(model, cfg_train.load_sdf_from, name='bg')
            elif cfg_train.get('load_bg_density', False):
                bg_density0 = utils.load_grid_data(model, cfg_train.load_sdf_from, name='bg_density', return_raw=True)
                model.init_bg_density_from_bg_density(bg_density0)
        else:
            model = utils.load_grid_data(model, cfg_train.load_sdf_from, name='sdf')
            smooth = getattr(model, 'init_sdf_smooth', False)
            if smooth:
                model.sdf = model.smooth_conv(model.sdf)
        # 这里原来没有加bending latenets list，但感觉应该是要加的，因为过完这个分支optimizer就被这个新的覆盖了
        # coarse没有进这个分支所以不是很影响
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0, bending_latents_list=model.bending_latents_list)
    # coarse和fine都没有进这个分支
    elif args.sdf_mode != "density" and load_density_from:
        smooth = getattr(model, 'init_density_smooth', True)
        model.init_sdf_from_density(smooth=smooth, reduce=1)
        # have to recreate the optimizer
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    # initial mesh evaluation
    # if stage == 'surf':
    #     gt_eval = 'dtu' in cfg.basedir
    #     validate_mesh(model, resolution=256, world_space=True, prefix="init", scale_mats_np=data_dict['scale_mats_np'], scene=args.scene, gt_eval=gt_eval)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    # 这是内部定义的一个函数，但是此时并不会运行，只有调用到它是才会执行
    def gather_training_rays():
        if data_dict['irregular_shape']:
            # 不规则形状
            '''
            coarse:
            [80, 540, 960, 3]
            [80, 540, 960, 1]
            fine:
            [72, 1080, 1920, 3]
            [72, 1080, 1920, 1]
            fine的时候不是用所有的图片train，且没有对图片进行放缩
            '''
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
            mask_tr_ori = [masks[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            # coarse和fine都走这个分支
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
            mask_tr_ori = masks[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            # fine阶段走这个分支，coarse在非init部分也走这个分支
            # 对应的阶段调对应的模型的成员函数
            # 执行该函数时所用显存显著增加，因为一下子存了所有的rays_o，rays_d等
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays_in_maskcache_sampling(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                model=model, render_kwargs=render_kwargs,
            )
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            # 自己的custom数据coarse阶段走这个分支
            # Model = lib.voxurf_coarse
            # 这里调的是voxurf_coarse.py中的函数
            # 返回的是所有图片所有像素点对应的rgb等信息
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = Model.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        if cfg_train.ray_sampler == 'patch':
            # coarse阶段未进这个分支
            # patch sampler contains lots of empty spaces, remove them.
            index_generator = Model.batch_indices_generator(len(rgb_tr), 1)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler
    #----函数定义结束-----##

    # 调用上述定义的函数
    # coarse后半部分和fine阶段这里返回的是筛选后的rgb，ray等
    # __import__('ipdb').set_trace()

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()
    # __import__('ipdb').set_trace()
    
    if cfg_train.pervoxel_lr:
        # coarse init的时候会进这个分支，初始化每个voxel网格
        # fine阶段不会再进了
        def per_voxel_init():
            cnt = model.voxel_count_views(
                rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.density[cnt <= 2] = -100
        per_voxel_init()

    # GOGO
    psnr_lst = []
    weight_lst = []
    mask_lst = []
    bg_mask_lst = []
    weight_sum_lst = []
    weight_nonzero_lst = []
    s_val_lst = []
    time0 = time.time()
    logger.info("start: {} end: {}".format(1 + start, 1 + cfg_train.N_iters))

    for global_step in trange(1+start, 1+cfg_train.N_iters):        
        # __import__('ipdb').set_trace()
        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            # coarse阶段不进该分支，fine阶段也没进
            if hasattr(model, 'num_voxels_bg'):
                model.scale_volume_grid(model.num_voxels * scale_ratio, model.num_voxels_bg * scale_ratio)
            else:
                model.scale_volume_grid(model.num_voxels * scale_ratio)
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

        optimizer.zero_grad(set_to_none=True)

        # 添加frame
        frame = 0

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            # coarse阶段init时不会进入该分支采样，后续会
            # fine阶段会进入这个分支采样
            # __import__('ipdb').set_trace()

            '''
            这里直接就拿到sel_i了，但并不是按照每帧来的
            目前还没看懂这里是的sel_i是根据什么得出来的
            '''
            # sel_i.shape = [8192]
            # 这里应该也要该成单张图采样试试
            '''
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            '''
            # sel_b是在imsz中的位置，并不是frame，imsz与i_train等长，可以以此获得frame
            sel_b = torch.randint(len(imsz), [1])
            frame = i_train[sel_b[0]]
            # 这张图有多少个有效采样点
            n = imsz[sel_b]
            # 通过sum得到起始点
            begin = sum(imsz[:sel_b])
            # 从该图的采样点中挑出8192个
            sel_b = torch.randint(begin, begin + n, [cfg_train.N_rand])
            target = rgb_tr[sel_b]
            rays_o = rays_o_tr[sel_b]
            rays_d = rays_d_tr[sel_b]
            viewdirs = viewdirs_tr[sel_b]
            


        elif cfg_train.ray_sampler == 'patch':
            
            # 只有这里没改了，加个断点看代码会不会进到这里
            # __import__('ipdb').set_trace()

            sel_b = batch_index_sampler()
            patch_size = cfg_train.N_patch
            sel_r_start = torch.randint(rgb_tr.shape[1] - patch_size, [1])
            sel_c_start = torch.randint(rgb_tr.shape[2] - patch_size, [1])
            sel_r, sel_c = torch.meshgrid(torch.arange(sel_r_start[0], sel_r_start[0] + patch_size),
                                          torch.arange(sel_c_start[0], sel_c_start[0] + patch_size))
            sel_r, sel_c = sel_r.reshape(-1), sel_c.reshape(-1)
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        elif cfg_train.ray_sampler == 'random':
            # coarse阶段首次会进入随机采样的分支
            '''
            长度都是8192 -- cfg_train.N_rand = 8192
            rgb_tr.shape = [80, 540, 940, 3]
            torch.randint: 在该范围内随机采样8192个点
            sel_b：选择哪张图片
            sel_r：宽像素点是几
            sel_c：长像素点是几
            sel_r和sel_c可以定位一张图片中的一个像素点

            下面的shape都是[8192, 3]
            '''

            #-------直接把这里改了可行吗？应该可行---------#
            # __import__('ipdb').set_trace()

            # sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            # 改成随机采样一帧，在该帧上采样8192个点

            # 检验一下dtu是不是进到这个分支, 是的
            # __import__('ipdb').set_trace()

            sel_b = torch.randint(rgb_tr.shape[0], [1])
            frame = sel_b[0]
            # 扩展维度
            sel_b = sel_b.expand(cfg_train.N_rand)
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]

        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            # 上设备
            # 都是 [8192, 3]
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        '''
        coarse阶段的init部分这里调DirectVoxGO()模型的forward函数
        
        zhj:
        # rays_o, requires_grad is True
        rays_o,rays_d, viewdirs, requires_grad is False
        '''
        render_result = model(rays_o, rays_d, viewdirs, frame, global_step=global_step, **render_kwargs)

        if global_step == cfg_train.N_iters:
            # 最后记一下隐向量的值
            with open('bending_latents.txt','w+') as f:
                str_latents = ",\n".join(["".join(str(latent.tolist())) for latent in model.bending_latents_list])
                f.write(str_latents)
        
        # __import__('ipdb').set_trace()

        # gradient descent step
        # optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach()).item()

        writer.add_scalar('train/rgb_loss', loss, global_step)

        # 计算loss
        # 加上bending network的loss
        if cfg.bending_network is not None:
            offset_loss = render_result['offset_loss']
            divergence_loss = render_result['divergence_loss']
            # print("=========offset weight: {}, divergence weight: {}=========".format(cfg_train.offset_weight, cfg_train.divergence_weight))
            bending_losses = offset_loss * cfg_train.offset_weight \
                            + divergence_loss * cfg_train.divergence_weight
            if cfg_train.bending_increasing:
                # 进了这个分支
                # Increasing schedule for bending losses as per NR-NeRF
                bending_losses *= ((1. / 100.) ** (1. - (global_step / cfg_train.N_iters)))
            loss += bending_losses

        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_cum'][...,-1].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss

        if cfg_train.weight_rgbper > 0:
            # coarse阶段走这个分支
            rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss += cfg_train.weight_rgbper * rgbper_loss

        if global_step>cfg_train.tv_from and global_step<cfg_train.tv_end and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                tv_terms = getattr(cfg_train, 'tv_terms', dict())
                sdf_tv, smooth_grad_tv = tv_terms['sdf_tv'], tv_terms['smooth_grad_tv']
                if smooth_grad_tv > 0:
                    loss += cfg_train.weight_tv_density * model.density_total_variation(sdf_tv=0, smooth_grad_tv=smooth_grad_tv)
                if getattr(cfg_train, 'ori_tv', False):
                    loss += cfg_train.weight_tv_density * model.density_total_variation(sdf_tv=sdf_tv, smooth_grad_tv=0)
                    weight_tv_k0 = getattr(cfg_train, 'weight_tv_k0')
                    if weight_tv_k0 > 0:
                        k0_tv_terms = getattr(cfg_train, 'k0_tv_terms', dict())
                        loss += cfg_train.weight_tv_k0 * model.k0_total_variation(**k0_tv_terms)
                    if getattr(tv_terms, 'bg_density_tv', 0):
                        loss += cfg_train.weight_tv_density * model.density_total_variation(sdf_tv=0, smooth_grad_tv=0, bg_density_tv=tv_terms['bg_density_tv'])
        if getattr(cfg_train, 'ori_tv', False) and cfg_train.get('weight_bg_tv_k0', 0) >0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0 and global_step<cfg_train.tv_end:
            bg_k0_tv_terms = getattr(cfg_train, 'bg_k0_tv_terms', dict())
            loss += cfg_train.get('weight_bg_tv_k0', 0) * model.bg_k0_total_variation(**bg_k0_tv_terms)

        if getattr(cfg_train, 'weight_rgb0', 0.) > 0:
            loss += F.mse_loss(render_result['rgb_marched0'], target) * cfg_train.weight_rgb0

        
        loss.backward(retain_graph=True)
        '''
        for key, param in model.bending_network.named_parameters():
            print(key, ' : ',param.grad)
        for key, param in model.named_parameters():
            print(key, ' : ',param.grad)
        '''
        
        # make sure that density has no grad
        if global_step>cfg_train.tv_from and global_step<cfg_train.tv_end and global_step%cfg_train.tv_every==0:
            if not getattr(cfg_train, 'ori_tv', False):
                if cfg_train.weight_tv_density>0:
                    tv_terms = getattr(cfg_train, 'tv_terms', dict())
                    sdf_tv = tv_terms['sdf_tv']
                    if sdf_tv > 0:
                        model.sdf_total_variation_add_grad(
                            cfg_train.weight_tv_density * sdf_tv / len(rays_o), global_step < cfg_train.tv_dense_before)
                    bg_density_tv = getattr(tv_terms, 'bg_density_tv', 0)
                    if bg_density_tv > 0:
                        model.bg_density_total_variation_add_grad(
                            cfg_train.weight_tv_density * bg_density_tv / len(rays_o), global_step < cfg_train.tv_dense_before)                                                                                   
                if cfg_train.weight_tv_k0 > 0:
                    model.k0_total_variation_add_grad(
                        cfg_train.weight_tv_k0 / len(rays_o), global_step < cfg_train.tv_dense_before)
                if getattr(cfg_train, 'weight_bg_tv_k0', 0) > 0:
                    model.bg_k0_total_variation_add_grad(
                        cfg_train.weight_bg_tv_k0 / len(rays_o), global_step < cfg_train.tv_dense_before)
        optimizer.step()
        wm = render_result['weights'].max(-1)[0]
        ws = render_result['weights'].sum(-1)
        if (wm>0).float().mean() > 0:
            psnr_lst.append(psnr)
            '''
            这些的长度都是1
            '''
            weight_lst.append(wm[wm>0].mean().detach().cpu().numpy())
            weight_sum_lst.append(ws[ws>0].mean().detach().cpu().numpy())
            weight_nonzero_lst.append((ws>0).float().mean().detach().cpu().numpy())
            mask_lst.append(render_result['mask'].float().mean().detach().cpu().numpy())
            if 'bg_mask' in render_result:
                bg_mask_lst.append(render_result['bg_mask'].float().mean().detach().cpu().numpy())
        s_val = render_result["s_val"] if "s_val" in render_result else 0
        s_val_lst.append(s_val)

        writer.add_scalar('train/psnr', psnr, global_step)
        writer.add_scalar('train/s_val', s_val, global_step)
        writer.add_scalar('train/mask', mask_lst[-1], global_step)
        writer.add_scalar('train/offset_loss', offset_loss, global_step)
        writer.add_scalar('train/divergence_loss', divergence_loss, global_step)
        writer.add_scalar('train/bending_losses', bending_losses, global_step)


        global_step_ = global_step - 1
        # update lr
        '''
        更新学习率
        像这种两个模型都有且只能留一个的部分得考察一下留谁的
        '''
        N_iters = cfg_train.N_iters
        if not getattr(cfg_train, 'cosine_lr', ''):
            decay_steps = cfg_train.lrate_decay * 1000
            decay_factor = 0.1 ** (1/decay_steps)
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = param_group['lr'] * decay_factor
        else:
            def cosine_lr_func(iter, warm_up_iters, warm_up_min_ratio, max_steps, const_warm_up=False, min_ratio=0):
                if iter < warm_up_iters:
                    if not const_warm_up:
                        lr = warm_up_min_ratio + (1 - warm_up_min_ratio) * (iter / warm_up_iters)
                    else:
                        lr = warm_up_min_ratio
                else:
                    lr = (1 + math.cos((iter - warm_up_iters) / (max_steps - warm_up_iters) * math.pi)) * 0.5 * (1 - min_ratio) + min_ratio
                return lr
            def extra_warm_up_func(iter, start_iter, warm_up_iters, warm_up_min_ratio):
                if iter >= start_iter:
                    extra_lr = warm_up_min_ratio + (1 - warm_up_min_ratio) * (iter - start_iter) / warm_up_iters
                    return min(extra_lr, 1.0)
                else:
                    return 1.0
            warm_up_iters = cfg_train.cosine_lr_cfg.get('warm_up_iters', 0) 
            warm_up_min_ratio = cfg_train.cosine_lr_cfg.get('warm_up_min_ratio', 1.0)
            const_warm_up = cfg_train.cosine_lr_cfg.get('const_warm_up', False)
            cos_min_ratio = cfg_train.cosine_lr_cfg.get('cos_min_ratio', False)
            if global_step == 0:
                pre_decay_factor = 1.0
            else:
                pre_decay_factor = cosine_lr_func(global_step_ - 1, warm_up_iters, warm_up_min_ratio, N_iters, const_warm_up, cos_min_ratio)
            pos_decay_factor = cosine_lr_func(global_step_, warm_up_iters, warm_up_min_ratio, N_iters, const_warm_up, cos_min_ratio)
            decay_factor = pos_decay_factor / pre_decay_factor
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = param_group['lr'] * decay_factor
        
        # __import__('ipdb').set_trace()

        decay_step_module = getattr(cfg_train, 'decay_step_module', dict())
        if global_step_ in decay_step_module:
            # __import__('ipdb').set_trace()
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                # 这里出了问题，没有'name'这个key，奇了怪了
                # 明天跑一下2080的代码对比找一下问题吧
                try:
                    '''
                    因为只有前四项有，后我们自己加的bending network以及隐向量目前对应的是没有的
                    '''
                    if param_group['name'] in decay_step_module[global_step_]:
                        decay_factor = decay_step_module[global_step_][param_group['name']]
                        param_group['lr'] = param_group['lr'] * decay_factor
                        logger.info('- '*10 + '[Decay lrate] for {} by {}'.format(param_group['name'], decay_factor) + ' -'*10)
                except:
                    break


        # update tv terms
        tv_updates = getattr(cfg_train, 'tv_updates', dict())
        if global_step_ in tv_updates:
            for tv_term, value in tv_updates[global_step_].items():
                setattr(cfg_train.tv_terms, tv_term, value)
            logger.info('- '*10 + '[Update tv]: ' + str(tv_updates[global_step_]) + ' -'*10)

        # update s_val func
        s_updates = getattr(cfg_model, 's_updates', dict())
        if global_step_ in s_updates:
            for s_term, value in s_updates[global_step_].items():
                setattr(model, s_term, value)
            logger.info('- '*10 + '[Update s]: ' + str(s_updates[global_step_]) + ' -'*10)

        # update smooth kernel
        smooth_updates = getattr(cfg_model, 'smooth_updates', dict())
        if global_step_ in smooth_updates:
            model.init_smooth_conv(**smooth_updates[global_step_])
            logger.info('- '*10 + '[Update smooth conv]: ' + str(smooth_updates[global_step_]) + ' -'*10)

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            bg_mask_mean = 0. if len(bg_mask_lst) == 0 else np.mean(bg_mask_lst)
            logger.info(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                        f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                        f'Wmax: {np.mean(weight_lst):5.2f} / Wsum: {np.mean(weight_sum_lst):5.2f} / W>0: {np.mean(weight_nonzero_lst):5.2f}'
                        f' / s_val: {np.mean(s_val_lst):5.2g} / mask\%: {100*np.mean(mask_lst):1.2f} / bg_mask\%: {100*bg_mask_mean:1.2f} '
                        f'Eps: {eps_time_str}')
            psnr_lst, weight_lst, weight_sum_lst, weight_nonzero_lst, mask_lst, bg_mask_lst, s_val_lst = [], [], [], [], [], [], []

        # validate image
        # args.i_validate = 10000，且在fine的阶段才行
        if global_step%args.i_validate==0 and global_step != cfg_train.N_iters and stage == 'surf' and 'fine' in args.sdf_mode:
            
            # pass
            # __import__('ipdb').set_trace()

            render_viewpoints_kwargs = {
                'model': model,
                'ndc': cfg.data.ndc,
                'render_kwargs': {
                    'near': data_dict['near'],
                    'far': data_dict['far'],
                    'bg': 1 if cfg.data.white_bkgd else 0,
                    'stepsize': cfg_model.stepsize,
                    'inverse_y': cfg.data.inverse_y,
                    'flip_x': cfg.data.flip_x,
                    'flip_y': cfg.data.flip_y,
                    'render_grad': True,
                    'render_depth': True,
                    'render_in_out': True,
                },
            }
            validate_image(cfg, stage, global_step, data_dict, render_viewpoints_kwargs, eval_all=cfg_train.N_iters==global_step)

        # validate mesh
        prefix = args.prefix + '_' if args.prefix else ''
        prefix += args.suffix + '_' if args.suffix else ''
        if 'eval_iters' in cfg_train and stage == 'surf':
            if global_step - start in cfg_train.eval_iters and stage == 'surf':
                gt_eval = 'dtu' in cfg.basedir
                cd = validate_mesh(model, resolution=256,
                                   prefix="{}{}_fine".format(prefix, global_step),
                                   gt_eval=gt_eval,
                                   world_space=True,
                                   scale_mats_np=data_dict['scale_mats_np'],
                                   scene=args.scene)

        # save checkpoints
        if global_step == cfg_train.N_iters:

            # __import__('ipdb').set_trace()

            # 这里是存档点，我们试试在这添加对隐向量和bending network参数的存储
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'MaskCache_kwargs': model.get_MaskCache_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bending_latents_list': model.bending_latents_list,
                # 这是个迭代器
                # 'bending_network_parameters': model.bending_network.named_parameters,
                'bending_network': model.bending_network
            }, last_ckpt_path)
            logger.info(f'scene_rep_reconstruction ({stage}): saved checkpoints at '+ last_ckpt_path)
            
            # print("+++++++++++++++in dvgo stage++++++++++++++")
            # for key, param in model.bending_network.named_parameters():
                # print(key, ' : ',param)

        # final mesh validation
        if global_step == cfg_train.N_iters and stage == 'surf' and 'fine' in args.sdf_mode:
            validate_mesh(model, 512, threshold=0.0, prefix="{}final".format(prefix), world_space=True,
                          scale_mats_np=data_dict['scale_mats_np'], gt_eval='dtu' in cfg.basedir, runtime=False, scene=args.scene)


def train(args, cfg, data_dict):

    # init
    logger.info('train: start')
    eps_time = time.time()

    '''
    将arg记录到args.txt中，以后所以以后可以直接通过这个文件查看各个args的值
    我们也可以通过这个来观察都有哪些config
    备份一份，这是因为在训练的时候并不知道哪个config是最优的，这样方便后续分析结果
    '''
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    if args.run_dvgo_init:
        # coarse阶段会经过这个分支
        # fine阶段不会经过这个分支
        # coarse geometry searching
        eps_coarse = time.time()
        xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        # 这里就开始训练了，1w次

        # __import__('ipdb').set_trace()

        scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
            xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
            data_dict=data_dict, stage='coarse', use_dvgo=True)
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        logger.info("+ "*10 + 'train: coarse geometry searching in' + eps_time_str + " +"*10 )

    coarse_expname = cfg.expname0 + '/coarse'
    # 这里定义了coarse_ckpt_path，所以fine阶段和coarse阶段都用load这个
    coarse_ckpt_path = os.path.join(cfg.basedir, coarse_expname, f'coarse_last.tar')
    logger.info("+ "*10 + 'coarse_expname' + coarse_expname + " +"*10)

    if args.no_dvgo_init:
        # for the w\o mask setting
        box_size_ = cfg.surf_train.get('box_size', 1.5)
        print(">>> box_size: ", box_size_)
        xyz_min_fine, xyz_max_fine = torch.tensor([-box_size_,-box_size_,-box_size_]).cuda(), torch.tensor([box_size_, box_size_, box_size_]).cuda()
    else:
        # coarse阶段scene_rep_reconstruction后也会进这个分支
        # fine阶段直接进这个分支
        # 是run.py中定义的函数
        # 从coarse阶段训练的结果中计算bbox
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
            model_class=dvgo_ori.DirectVoxGO, model_path=coarse_ckpt_path,
            thres=cfg.fine_model_and_render.bbox_thres)

    # __import__('ipdb').set_trace()

    if hasattr(cfg, 'surf_train'):

        # 我们先把coarse init阶段调通，最起码loss不要上升
        # __import__('ipdb').set_trace()
        # coarse阶段也会进这个分支
        # fine阶段进入这个分支
        eps_surf = time.time()
        # 还是同样的函数，不过stage之类的改了
        scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.surf_model_and_render, cfg_train=cfg.surf_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='surf',
            coarse_ckpt_path=coarse_ckpt_path)
        eps_surf = time.time() - eps_surf
        eps_time_str = f'{eps_surf//3600:02.0f}:{eps_surf//60%60:02.0f}:{eps_surf%60:02.0f}'
        logger.info("+ "*10 + 'train: fine detail reconstruction in' + eps_time_str + " +"*10 )

        eps_time = time.time() - eps_time
        eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
        logger.info('train: finish (eps time' + eps_time_str + ')')

def validate_image(cfg, stage, step, data_dict, render_viewpoints_kwargs, eval_all=True):
    testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{stage}')
    os.makedirs(testsavedir, exist_ok=True)

    # __import__('ipdb').set_trace()

    # 这个应该是选出来render的frame吧,frame可以加到render_viewpoints_kwargs中，传给model
    rand_idx = random.randint(0, len(data_dict['poses'][data_dict['i_train']])-1)
    # add frame
    # rand_idx = 85
    frame = data_dict['i_train'][rand_idx]

    logger.info("validating test set idx: {}".format(rand_idx))

    eval_lpips_alex = args.eval_lpips_alex and eval_all
    eval_lpips_vgg = args.eval_lpips_alex and eval_all
    rgbs, disps = render_viewpoints(
        render_poses=data_dict['poses'][data_dict['i_train']][rand_idx][None],
        HW=data_dict['HW'][data_dict['i_train']][rand_idx][None],
        Ks=data_dict['Ks'][data_dict['i_train']][rand_idx][None],
        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']][rand_idx][None],
        masks=[data_dict['masks'][i].cpu().numpy() for i in data_dict['i_train']][rand_idx][None],
        savedir=testsavedir,
        eval_ssim=args.eval_ssim, eval_lpips_alex=eval_lpips_alex, eval_lpips_vgg=eval_lpips_vgg, idx=rand_idx, step=step,
        **render_viewpoints_kwargs, frame = frame)


def validate_mesh(model, resolution=128, threshold=0.0, prefix="", world_space=False,
                  scale_mats_np=None, gt_eval=False, runtime=True, scene=122, smooth=True,
                  extract_color=False):
    
    # __import__('ipdb').set_trace()

    os.makedirs(os.path.join(cfg.basedir, cfg.expname, 'meshes'), exist_ok=True)
    bound_min = model.xyz_min.clone().detach().float()
    bound_max = model.xyz_max.clone().detach().float()

    gt_path = os.path.join(cfg.data.datadir, "stl_total.ply") if gt_eval else ''
    vertices0, triangles = model.extract_geometry(bound_min, bound_max, resolution=resolution,
                                                  threshold=threshold, scale_mats_np=scale_mats_np,
                                                  gt_path=gt_path, smooth=smooth,
                                                  )

    if world_space and scale_mats_np is not None:
        vertices = vertices0 * scale_mats_np[0, 0] + scale_mats_np[:3, 3][None]
    else:
        vertices = vertices0

    if extract_color:
        # use normal direction as the viewdir
        ray_pts = torch.from_numpy(vertices0).cuda().float().split(8192 * 32, 0)
        vertex_colors = [model.mesh_color_forward(pts) for pts in ray_pts]
        vertex_colors = (torch.concat(vertex_colors).cpu().detach().numpy() * 255.).astype( np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
    else:
        mesh = trimesh.Trimesh(vertices, triangles)
    
    # __import__('ipdb').set_trace()

    mesh_path = os.path.join(cfg.basedir, cfg.expname, 'meshes', "{}_".format(scene)+prefix+'.ply')
    mesh.export(mesh_path)
    logger.info("mesh saved at " + mesh_path)
    if gt_eval:
        mean_d2s, mean_s2d, over_all = dtu_eval.eval(mesh_path, scene=scene, eval_dir=os.path.join(cfg.basedir, cfg.expname, 'meshes'),
                      dataset_dir='data/DTU', suffix=prefix+'eval', use_o3d=False, runtime=runtime)
        res = "standard point cloud sampling" if not runtime else "down sampled point cloud for fast eval (NOT standard!):"
        logger.info("mesh evaluation with {}".format(res))
        logger.info(" [ d2s: {:.3f} | s2d: {:.3f} | mean: {:.3f} ]".format(mean_d2s, mean_s2d, over_all))
        return over_all
    return 0.

if __name__=='__main__':
    # load setup

    # 准备参数
    # __import__('ipdb').set_trace()

    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # __import__('ipdb').set_trace()
    
    # reset the root by the scene id

    # __import__('ipdb').set_trace()

    if args.scene:
        cfg.expname += "{}".format(args.scene)
        # 这里有bug
        cfg.data.datadir += "{}".format(args.scene)
    cfg.expname0 = cfg.expname
    cfg.expname = cfg.expname + '/' + cfg.exp_stage
    if args.suffix:
        cfg.expname += "_" + args.suffix
    cfg.load_expname = args.load_expname if args.load_expname else cfg.expname
    # set up tensorboard
    # ./logs/custom/RealCactus/logs_all/RealCactus/coarse
    # 我们改一下，用数据集和case名来区分吧

    # __import__('ipdb').set_trace()

    # writer_dir = os.path.join(cfg.basedir, cfg.expname0, 'logs_all', cfg.expname)
    # 这样改一下
    writer_dir_new = os.path.join(cfg.basedir, cfg.expname0, "logs_all", args.prefix, cfg.exp_stage)

    print("---------------writer_dir: {}----------------".format(writer_dir_new))
    
    writer = SummaryWriter(log_dir=writer_dir_new)
    # set up the logger and tensorboard
    # ./logs/custom
    cfg.basedir0 = cfg.basedir
    if args.prefix:
        '''test'''
        cfg.basedir = os.path.join(cfg.basedir, args.prefix)
    log_dir = os.path.join(cfg.basedir, cfg.expname, 'log')
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now()
    time_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    logger = get_root_logger(logging.INFO, handlers=[
        logging.FileHandler(os.path.join(log_dir, '{}_train.log').format(time_str))])
    logger.info("+ "*10 + cfg.expname + " +"*10)
    logger.info("+ "*10 + log_dir + " +"*10)
    # set white or black color
    if cfg.get('use_sp_color', False):
        assert 'white_list' in cfg and 'black_list' in cfg
        if int(args.scene) in cfg['white_list']:
            assert args.scene not in cfg['black_list']
            cfg.data.white_bkgd = True
            logger.info("+ "*10 + str(args.scene) + ' white bg '  + " +"*10)
        if int(args.scene) in cfg['black_list']:
            assert args.scene not in cfg['white_list']
            cfg.data.white_bkgd = False
            logger.info("+ "*10 + str(args.scene) + ' black bg '  + " +"*10)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    seed_everything()
    if getattr(cfg, 'load_expname', None) is None:
        cfg.load_expname = args.load_expname if args.load_expname else cfg.expname
    logger.info(cfg.load_expname)
    os.makedirs(os.path.join(cfg.basedir, cfg.expname, 'recording'), exist_ok=True)
    if not args.render_only or args.mesh_from_sdf:
        copyfile('run.py', os.path.join(cfg.basedir, cfg.expname, 'recording', 'run.py'))
        copyfile(args.config, os.path.join(cfg.basedir, cfg.expname, 'recording', args.config.split('/')[-1]))
    import lib.dvgo_ori as dvgo_ori

    if args.sdf_mode == "voxurf_coarse":

        # __import__('ipdb').set_trace()

        import lib.voxurf_coarse as Model
        copyfile('lib/voxurf_coarse.py', os.path.join(cfg.basedir, cfg.expname, 'recording','voxurf_coarse.py'))
    elif args.sdf_mode == "voxurf_fine":
        import lib.voxurf_fine as Model
        
        # __import__('ipdb').set_trace()

        copyfile('lib/voxurf_fine.py', os.path.join(cfg.basedir, cfg.expname, 'recording','voxurf_fine.py'))
    elif args.sdf_mode == "voxurf_womask_coarse":
        import lib.voxurf_womask_coarse as Model
        copyfile('lib/voxurf_womask_coarse.py', os.path.join(cfg.basedir, cfg.expname, 'recording','voxurf_womask_coarse.py'))
    elif args.sdf_mode == "voxurf_womask_fine":
        import lib.voxurf_womask_fine as Model
        copyfile('lib/voxurf_womask_fine.py', os.path.join(cfg.basedir, cfg.expname, 'recording','voxurf_womask_fine.py'))
    else:
        raise NameError
    
    # __import__('ipdb').set_trace()
    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    '''
    此时的near和far是正常的
    ipdb> near
    0.13697632551193237
    ipdb> far
    2.7395265

    images : [80, 540, 960, 3]
    masks : [80, 540, 960, 1]
    '''

    # __import__('ipdb').set_trace()

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        # fine阶段没有进这个分支
        logger.info('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            # 循环，给每个图片的每个点都生成相应的rays
            rays_o, rays_d, viewdirs = Model.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
                            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
                            cam_lst=np.array(cam_lst))
        logger.info('done')
        sys.exit()

    if args.mesh_from_sdf:
        logger.info('Extracting mesh from sdf...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'surf_last.tar')
            if os.path.exists(ckpt_path):
                new_kwargs = cfg.surf_model_and_render
            else:
                ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
                new_kwargs = cfg.fine_model_and_render
            model = utils.load_model(Model.Voxurf, ckpt_path, new_kwargs).to(device)
            prefix = args.prefix + '_' if args.prefix else ''
            prefix += args.suffix + '_' if args.suffix else ''
            gt_eval = 'dtu' in cfg.basedir
            validate_mesh(model, 512, threshold=0.0, prefix="{}final_mesh".format(prefix), world_space=True,
                          scale_mats_np=data_dict['scale_mats_np'], gt_eval=gt_eval, runtime=False, scene=args.scene, extract_color=args.extract_color)
        logger.info('done')
        sys.exit()

    # train
    if not args.render_only:
        # 开始训练
        # 只要不是render_only就会开始train，fine阶段也一样，因为fine阶段也是要train的
        # coarse和fine都会开始train
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video or args.interpolate:
        
        # 这里应该是最后了

        # __import__('ipdb').set_trace()

        if args.ft_path:
            ckpt_path = args.ft_path
            new_kwargs = cfg.fine_model_and_render
        elif hasattr(cfg, 'surf_train'):
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'surf_last.tar')
            new_kwargs = cfg.surf_model_and_render
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
            new_kwargs = cfg.fine_model_and_render
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        print(">>> Loading from {}".format(ckpt_path))
        model = utils.load_model(Model.Voxurf, ckpt_path, new_kwargs).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_grad': True,
                'render_depth': True,
                'render_in_out': True,
            },
        }

    if args.interpolate:
        img_idx_0, img_idx_1 = args.interpolate.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        savedir = os.path.join(cfg.basedir, cfg.expname, f'interpolate_{img_idx_0}_{img_idx_1}')
        interpolate_view(savedir, img_idx_0, img_idx_1,
                         render_poses=data_dict['poses'],
                         HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                         Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                         render_factor=args.render_video_factor,
                         **render_viewpoints_kwargs
                         )

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_train']],
            HW=data_dict['HW'][data_dict['i_train']],
            Ks=data_dict['Ks'][data_dict['i_train']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
            masks=data_dict['masks'],
            savedir=testsavedir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
            masks=[data_dict['masks'][i].cpu().numpy() for i in data_dict['i_test']],
            savedir=testsavedir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render video
    if args.render_video:
        assert 'dtu' not in cfg.basedir, 'please try --interpolate for the DTU dataset.'
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
            render_poses=torch.from_numpy(data_dict['render_poses']).cuda(),
            HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            render_factor=args.render_video_factor,
            savedir=testsavedir,
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    logger.info('Done')
