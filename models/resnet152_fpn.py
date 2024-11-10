import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
import torch.nn as nn
from models.net import FPN_4 as FPN_4
from models.net import SSH as SSH
from models.heads import ClassHead as ClassHead
from models.heads import BboxHead as BboxHead
from models.heads import LandmarkHead as LandmarkHead

class Resnet152_fpn(nn.Module):
    def __init__(self, return_layers=None, cfg = None, phase = None):
        super(Resnet152_fpn, self).__init__()
        self.phase = phase

        backbone = models.resnet152(pretrained=cfg['pretrain'])
        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 1,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']

        self.c6_3x3 = nn.Conv2d(in_channels_list[3], out_channels, kernel_size = 3, stride = 2, padding=1)

        self.fpn_4 = FPN_4(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.ssh4 = SSH(out_channels, out_channels)
        self.ssh5 = SSH(out_channels, out_channels)


        self.ClassHead = self._make_class_head(fpn_num=5, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=5, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=5, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        out = list(out.values())
        
        # print(inp[0].shape)
        # print(inp[1].shape)
        # print(inp[2].shape)
        # print(inp[3].shape)
        c6_3x3 = self.c6_3x3(out[3])
        # print(c6_3x3.shape)
        # FPN
        fpn = self.fpn_4(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        feature4 = self.ssh4(fpn[3])
        feature5 = self.ssh5(c6_3x3)

        features = [feature1, feature2, feature3, feature4, feature5]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

        