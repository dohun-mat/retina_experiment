import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from models.net import BiFPN as BiFPN
from models.net import SSH as SSH
from models.heads import ClassHead as ClassHead
from models.heads import BboxHead as BboxHead
from models.heads import LandmarkHead as LandmarkHead


class Resnet50_bifpn(nn.Module):
    def __init__(self, cfg = None, phase = None):
        super(Resnet50_bifpn, self).__init__()

        self.phase = phase
        backbone = models.resnet50(pretrained=True)
        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8
        ]
        out_channels = cfg['out_channel']

        conv_channel_coef = {
                0: [in_channels_list[0], in_channels_list[1], in_channels_list[2]],
            }
        self.fpn_num_filters = out_channels
        self.fpn_cell_repeats = 3
        self.compound_coef=0
        self.bifpn = nn.Sequential(
        *[BiFPN(self.fpn_num_filters,
                conv_channel_coef[self.compound_coef],
                True if _ == 0 else False,
                attention=True if self.compound_coef < 6 else False,
                use_p8=self.compound_coef > 7)
            for _ in range(self.fpn_cell_repeats)])
        

        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

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
    
    def forward(self, x):
        out = self.body(x)
        out = list(out.values())

        bifpn = self.bifpn(out)
        feature1 = self.ssh1(bifpn[0])
        feature2 = self.ssh2(bifpn[1])
        feature3 = self.ssh3(bifpn[2])

        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
