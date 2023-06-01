import math
import numpy as np

from tinygrad.tensor import Tensor
import tinygrad.nn as nn
import tinygrad.mlops as mlops
import extra.functional as F
from extra.utils import get_child,AnchorGenerator,ImageTransforms
from models.rpn import RPNHead, RegionProposalNetwork
from extra.pooler import RoIAlign
from extra.roiheads import RoIHeads
from models.resnet import ResNet
from collections import OrderedDict


class MaskRCNN:
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        #RPN
        anchor_sizes = (128, 256, 512)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #RoIHeads
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)
        
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        
        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)
        
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        
        #Image Transformer
        self.transformer = ImageTransforms(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
    
    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]
        
        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]
        feature = self.backbone(image)
        
        proposal, rpn_losses = self.rpn(feature, image_shape, target)
        result, roi_losses = self.head(feature, proposal, image_shape, target)
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result

class FastRCNNPredictor:
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)
        return score, bbox_delta     

class MaskRCNNPredictor:
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        self.mask_layers = []
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            conv = nn.Conv2d(next_feature, layer_features, kernel_size=3, stride=1, padding=1)
            relu = F.ReLU(inplace=True)
            self.mask_layers.append(conv)
            self.mask_layers.append(relu)
            next_feature = layer_features

        self.mask_conv5 = nn.ConvTranspose2d(next_feature, dim_reduced, kernel_size=2, stride=2, padding=0)
        self.relu5 = F.ReLU(inplace=True)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.mask_layers:
            if isinstance(layer, nn.Conv2d):
                layer = F.kaiming_normal_init(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        for layer in self.mask_layers:
            x = layer(x)

        x = self.mask_conv5(x)
        x = self.relu5(x)
        x = self.mask_fcn_logits(x)

        return x

class ResBackbone:
    def __init__(self, backbone_name, pretrained):
        super().__init__()
                
        self.body = ResNet
        in_channels = 2048
        self.out_channels = 256
        
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x
    
def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    
    if pretrained:
        backbone_pretrained = False
        
    backbone = ResBackbone('resnet50', pretrained_backbone)
    model = MaskRCNN(backbone, num_classes)
    
    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])
        
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pretrained_msd[i])
            
        model.load_state_dict(msd)
    
    return model