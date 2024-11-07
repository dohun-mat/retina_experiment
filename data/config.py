# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re_bifpn = {
    'name': 'resnet50_bifpn',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 3,
    'epoch': 300,
    'decay1': 70,
    'decay2': 90,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

cfg_re_fpn = {
    'name': 'resnet50_fpn',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 3,
    'epoch': 300,
    'decay1': 70,
    'decay2': 90,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

cfg_cnv2_tiny = {
    'name': 'convnet_v2_tiny',
    'min_sizes': [[16, 25.40], [32, 50.8], [64, 101.59], [128, 203.19], [256,406.37]],
    'steps': [4, 8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 12,
    'ngpu': 3,
    'epoch': 300,
    'decay1': 70,
    'decay2': 90,
    'image_size': 640,
    'pretrain': False,
    # 'return_layers' : {'0': 'stage_0', '1': 'stage_1', '2': 'stage_2', '3': 'stage_3'},
    'return_layers' : {'stages_0' : 0, 'stages_1' : 1, 'stages_2' :2, 'stages_3' :3},
    'in_channel': 128,
    'out_channel': 256
}

cfg_cspres50_bifpn = {
    'name': 'cspresnet50_bifpn',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 1,
    'epoch': 300,
    'decay1': 70,
    'decay2': 90,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'CrossStage0': 1, 'CrossStage1': 2, 'CrossStage2': 3, 'CrossStage3': 4},
    'in_channel': 128,
    'out_channel': 256
}