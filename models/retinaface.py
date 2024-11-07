import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mobile_bifpn import mobile_bifpn as mobile_bifpn
from models.resnet50_FPN import Resnet50_fpn as Resnet50_fpn
from models.resnet50_bifpn import Resnet50_bifpn as Resnet50_bifpn
from models.cspresnet50_bifpn import cspResnet50_bifpn as cspResnet50_bifpn

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        print(cfg)
        super(RetinaFace,self).__init__()

        if cfg['name'] == 'mobile0.25':
            self.model_result = mobile_bifpn(cfg=cfg, phase = phase)
        elif cfg['name'] == 'resnet50_fpn':
            self.model_result = Resnet50_fpn(cfg=cfg, phase = phase)
        elif cfg['name'] == 'resnet50_bifpn':
            self.model_result = Resnet50_bifpn(cfg=cfg, phase = phase)
        elif cfg['name'] == 'cspresnet50_bifpn':
            self.model_result = cspResnet50_bifpn(cfg=cfg, phase = phase)
            
    def forward(self, x):
        return self.model_result(x)