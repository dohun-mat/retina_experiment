from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.data import RandomSampler
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, split_dataset, preproc 
from data import cfg_mnet, cfg_re_bifpn, cfg_re_fpn, cfg_cnv2_tiny, cfg_cspres50_bifpn, cfg_res152_fpn
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
import subprocess
import wandb
from sklearn.model_selection import train_test_split
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def test(epoch, args, cfg, net, test_dataset, criterion, val_loader, max_iter, priors):
    net.eval()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            load_t0 = time.time()
            
            images = images.cuda(non_blocking=True)
            targets = [anno.cuda(non_blocking=True) for anno in targets]

            # Forward
            out = net(images)
            
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            
            # ETA 및 학습 속도 계산
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - epoch) * len(val_loader))
            

            #################################################################
            # 출력 및 WandB 로깅
            if int(os.environ['LOCAL_RANK']) == 0:
            #############################################################
                print(f'test || Epoch: {epoch}/{max_iter} || Batch: {batch_idx+1}/{len(val_loader)} || Loc : {loss_l.item():.4f} Cla : {loss_c.item():.4f} Landm : {loss_landm.item():.4f} Total_Loss: {loss.item():.4f} || batch_time : {batch_time:.4f} || ETA : {str(datetime.timedelta(seconds=eta))}')
                wandb.log({"test_epoch": epoch, "test_total_loss": loss.item(), "test_loss_l": loss_l.item(),
                            "test_loss_c": loss_c.item(), "test_loss_landm": loss_landm.item()})
                    
    
def adjust_learning_rate(args, optimizer, gamma, epoch, step_index, iteration, epoch_size):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        warmup_epoch = -1
        if epoch <= warmup_epoch:
            lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
        else:
            lr = args.lr * (gamma ** (step_index))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_model(max_iter, epoch, net, data_loader, optimizer, criterion, cfg, args, stepvalues, step_index, priors):
    net.train()  # 모델을 학습 모드로 설정
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        
        if epoch in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(args = args, optimizer = optimizer, gamma = args.gamma, epoch = epoch, step_index = step_index, iteration = epoch*len(data_loader), epoch_size = len(data_loader))
        
        load_t0 = time.time()
        
        images = images.cuda(non_blocking=True)
        targets = [anno.cuda(non_blocking=True) for anno in targets]

        # print('images_shape')
        # print(images.shape)
        # Forward
        out = net(images)

        # Backward e otimização
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()

        
        # ETA 및 학습 속도 계산
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - epoch) * len(data_loader))
        lr = optimizer.param_groups[0]['lr']
        
        ######################################################################
        # 출력 및 WandB 로깅
        if int(os.environ['LOCAL_RANK']) == 0:
        ##################################################################
            print(f'train || Epoch: {epoch}/{max_iter} || Batch: {batch_idx+1}/{len(data_loader)} || Loc : {loss_l.item():.4f} Cla : {loss_c.item():.4f} Landm : {loss_landm.item():.4f} Total_Loss: {loss.item():.4f} || LR: {lr} || batch_time : {batch_time:.4f} || ETA : {str(datetime.timedelta(seconds=eta))}')
            wandb.log({"train_epoch": epoch, "train_total_loss": loss.item(), "train_loss_l": loss_l.item(),
                        "train_loss_c": loss_c.item(), "train_loss_landm": loss_landm.item()})
    # 학습률 업데이트
    # scheduler.step()

def start(args):
    # 현재 날짜를 "YYYY-MM-DD" 형식으로 포맷팅
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    wandb.init(project=f'face_detection')
    wandb.run.name = f'{args.network}_current_date wandb'
    wandb.run.save()
    wandb.config.update(args)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50_fpn":
        cfg = cfg_re_fpn
    elif args.network == "resnet50_bifpn":
        cfg = cfg_re_bifpn
    elif args.network == "cspresnet50_bifpn":    
        cfg = cfg_cspres50_bifpn
    elif args.network == "resnet152_fpn":    
        cfg = cfg_res152_fpn

        
    rgb_mean = (104, 117, 123) # bgr order
    num_classes = 2
    img_dim = cfg['image_size']
    num_gpu = cfg['ngpu']
    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    gpu_train = cfg['gpu_train']

    num_workers = args.num_workers
    momentum = args.momentum
    weight_decay = args.weight_decay
    initial_lr = args.lr
    # gamma = args.gamma
    training_dataset = args.training_dataset
    test_dataset = args.test_dataset
    save_folder = args.save_folder

    net = RetinaFace(cfg=cfg)
    print("Printing net...")
    # print(net)

    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load(args.resume_net)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    if num_gpu > 1 and gpu_train:
        #############################################################################
        local_rank = int(os.environ['LOCAL_RANK'])
        net = net.cuda(local_rank)
        net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)

        # net = net.to('cuda:1')
        # net = torch.nn.DataParallel(net).cuda()
        ###############################################################################
    else:
        net = net.cuda()

    cudnn.benchmark = True

    # optimizer = optim.AdamW(net.parameters(), lr=initial_lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    
    # net.train()
    start_epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    max_iter = max_epoch
    if args.resume_epoch > 0:
        start_epoch = args.resume_epoch
    else:
        start_epoch = 0

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    train_set, val_set = split_dataset(dataset, split_ratio=0.8)
        
    # 시드가 지정된 generator 생성
    train_generator = torch.Generator().manual_seed(42)
    val_generator = torch.Generator().manual_seed(42)

#################################################################
    sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=True)
    
    # sampler = RandomSampler(train_set)
    # val_sampler = RandomSampler(val_set)
#################################################################
    data_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, collate_fn=detection_collate, sampler = sampler, generator = train_generator)
    val_loader = data.DataLoader(val_set, batch_size=16, num_workers=num_workers, collate_fn=detection_collate, sampler = val_sampler, generator = val_generator)  
    
    stepvalues = (cfg['decay1'] , cfg['decay2'])
    step_index = 0 


    for epoch in range(start_epoch, max_iter+1):
        train_model(args = args, priors = priors, max_iter = max_iter, epoch = epoch, net = net, data_loader = data_loader, optimizer = optimizer, criterion = criterion, cfg = cfg, stepvalues = stepvalues, step_index = step_index)

        # 지정된 주기마다 테스트 평가
        if epoch % 10 == 0:
            test(args = args, priors = priors, epoch = epoch, net = net, test_dataset = test_dataset, criterion = criterion, cfg = cfg, val_loader = val_loader, max_iter = max_iter)

        # 모델 체크포인트 저장
        if epoch % 10 == 0:
            torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--training_dataset', default='/data/dhk/data/widerface/train/label.txt', help='Training dataset directory')
    parser.add_argument('--test_dataset', default='/data/dhk/data/widerface/val/wider_val.txt', type=str, help='dataset path')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--num_workers', default=32, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    args = parser.parse_args()

#####################################################################################################################################
      # print("~~~~~~~")
    os.environ['MASTER_ADDR'] = '223.195.111.29'
    os.environ['MASTER_PORT'] = '12012'

    # 환경 변수 설정 확인
    print(f'RANK: {os.environ.get("RANK")}, WORLD_SIZE: {os.environ.get("WORLD_SIZE")}, LOCAL_RANK: {os.environ.get("LOCAL_RANK")}')

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    print("Initializing process group...")
    
    dist.init_process_group(backend='nccl', rank=rank, init_method='env://')
    torch.cuda.set_device(local_rank)
########################################################################################################################################
    start(args)
##################################################
    dist.destroy_process_group()
###################################################################################