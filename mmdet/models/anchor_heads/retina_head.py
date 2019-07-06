import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from .anchor_head import AnchorHead
from ..registry import HEADS
from ..utils import bias_init_with_prob


@HEADS.register_module
class RetinaHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_images_3dce=1,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 bn_to_head=False,
                 **kwargs):
        self.num_images_3dce = num_images_3dce
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.bn_to_head = bn_to_head
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_scales=anchor_scales,
            use_sigmoid_cls=True,
            use_focal_loss=True,
            **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        if self.bn_to_head:
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(nn.Sequential(
                    nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(self.feat_channels),
                    nn.ReLU(inplace=True)
                ))
                self.reg_convs.append(nn.Sequential(
                    nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(self.feat_channels),
                    nn.ReLU(inplace=True)
                ))
        else:
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
                self.reg_convs.append(
                    nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x

        if not self.bn_to_head:
            for cls_conv in self.cls_convs:
                cls_feat = self.relu(cls_conv(cls_feat))
            for reg_conv in self.reg_convs:
                reg_feat = self.relu(reg_conv(reg_feat))
        else:
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

        # for 3dce
        if self.num_images_3dce != 1:
            bs, c, w, h = cls_feat.size()
            imgs_per_gpu = int(bs / self.num_images_3dce)
            cls_feat = cls_feat.view(imgs_per_gpu, -1, w, h)
            reg_feat = reg_feat.view(imgs_per_gpu, -1, w, h)

        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
