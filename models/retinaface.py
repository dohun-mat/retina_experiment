import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict

from models.net import MobileNetV1 as MobileNetV1
# from models.net import FPN as FPN
from models.net import SSH as SSH
# from models.net import spread_FPN as spread_FPN
from models.net import BiFPN as BiFPN

class IntermediateLayerGetter_conv(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        # if not set(return_layers).issubset([name for name, _ in model.named_children()]):
        #     raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name_1, module in model.named_children():
            for name_2, sub_module in module.named_children():
                full_name = f"{name_1}_{name_2}"
                # print("full_name")
                # print(full_name)
                layers[full_name] = sub_module
                if full_name in return_layers:
                    del return_layers[full_name]
                if not return_layers:
                    break
            if not return_layers:
                break

        super(IntermediateLayerGetter_conv, self).__init__(layers)
        self.return_layers = orig_return_layers
    
    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            # print("name")
            # print(name)
            x = module(x)
            if name in self.return_layers:
                # print("##############")
                # print(name)
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        
        elif cfg['name'] == 'convnet_v2_tiny':
            # print("~~~bbbbbbbbbbbbbbbbbbbbbbbbbbbb")
            import timm
            backbone = timm.create_model('convnextv2_base', pretrained=True)
            self.body = IntermediateLayerGetter_conv(backbone, cfg['return_layers'])
        elif cfg['name'] == 'cspresnet50':
            # print("~~~bbbbbbbbbbbbbbbbbbbbbbbbbbbb")
            import timm
            self.body = timm.create_model('cspresnet50', pretrained=True, features_only=True, out_indices=(2, 3, 4))
            


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
    
        # self.spread_fpn  = spread_FPN(in_channels_list,out_channels)
        # self.fpn = FPN(in_channels_list,out_channels)
        # self.bifpn = BiFPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        # self.ssh4 = SSH(out_channels, out_channels)
        # self.ssh5 = SSH(out_channels, out_channels)

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

    def forward(self,inputs):
        model_out = self.body(inputs)
        
        # out = list(out.values())
        # output4 = self.output4(input[3])
        
        # print(len(out))
        # print('out')
        # print(out[0].shape)
        # print(model_out[1].shape)
        # print(model_out[2].shape)
        # print(model_out[3].shape)

        out = [model_out[1], model_out[2], model_out[3]] 
        # FPN
        # fpn = self.fpn(out)
        bifpn = self.bifpn(out)
        # SSH
        feature1 = self.ssh1(bifpn[0])
        feature2 = self.ssh2(bifpn[1])
        feature3 = self.ssh3(bifpn[2])
        # feature4 = self.ssh4(bifpn[3])
        # feature5 = self.ssh5(bifpn[4])
        
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output