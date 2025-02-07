import os

_base_ = os.path.join('..', 'default_fine_s.py')

expname = ''
basedir = os.path.join('.', 'logs', 'custom')
train_all = True
reso_level = 2
exp_stage = 'coarse'
bending_network = True

data = dict(
    datadir=os.path.join('.', 'data', ''),
    dataset_type='dtu',
    inverse_y=True,
    white_bkgd= False
)

surf_train=dict(
    load_density_from='',

    pg_filter=[1000,],
    
    tv_add_grad_new=True,
    ori_tv=True,
    weight_main=1,    # this is for rgb_add
    N_iters=10000, #10000,
    lrate_decay=20,
    weight_tv_k0=0.01,
    weight_tv_density=0.001,
    tv_terms=dict(
        sdf_tv=0.1,
        grad_tv=0,
        smooth_grad_tv=0.05,
    ),
    tv_updates={
        1000:dict(
            sdf_tv=0.1,
            # grad_tv=10,
            smooth_grad_tv=0.2
        ),
    },
    tv_dense_before=20000,

    lrate_sdf=0.1,
    decay_step_module={
        1000:dict(sdf=0.1),
        5000:dict(sdf=0.5),
    },

    lrate_k0=1e-1, #1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3, # 1e-3,           # lr of the mlp to predict view-dependent color
    lrate_rgb_addnet=1e-3, # 1e-3,

    # bending network 相关的参数都先加到这里 (都放一份，暂时不知道应该放哪)
    bending_network=dict(
        latent_dim = 64,
        d_hidden = 64,
        n_layers = 5,
    ),

    bending_increasing = True,
    zero_init = True,
    flow_weight = 0.0, 
    divergence_weight = 200.0, # 参数还是要继续调的，这里先直接抄过来
    offset_weight = 20000.0
)

surf_model_and_render=dict(
    num_voxels=96**3,
    num_voxels_base=96**3,
    rgbnet_full_implicit=False, # by using a full mlp without local feature for rgb, the info for the geometry would be better
    posbase_pe=5,
    viewbase_pe=1,
    add_posbase_pe=5,
    add_viewbase_pe=4,
    rgb_add_res=True,
    rgbnet_depth=3,
    geo_rgb_dim=3,

    smooth_ksize=5,
    smooth_sigma=0.8,
    s_ratio=50,
    s_start=0.2,

    # bending network 相关的参数都先加到这里
    bending_network=dict(
        latent_dim = 64,
        d_hidden = 64,
        n_layers = 5,
    ),

    bending_network_train=dict(
        bending_increasing = True,
        zero_init = True,
        flow_weight = 0.0, 
        divergence_weight = 200.0, # 参数还是要继续调的，这里先直接抄过来
        offset_weight = 20000.0
    ),
)
