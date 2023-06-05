import inspect
import numpy as np
import re
from tinygrad.tensor import Tensor
import tinygrad.nn as nn
import tinygrad.mlops as mlops
import extra.functional as F
from extra.utils import get_child,AnchorGenerator,ImageTransforms,download_file,fake_torch_load
from models.rpn import RPNHead, RegionProposalNetwork
from extra.pooler import RoIAlign
from extra.roiheads import RoIHeads
from models.resnet import ResNet
from models.fpn import FPN
from collections import OrderedDict
from pathlib import Path
from typing import Tuple,List,Optional,Union

class MaskRCNN:
    def __init__(self, backbone, num_classes, box_head,
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
        num_anchors = len(anchor_sizes)
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
        
        self.roi_heads = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        

        
        layers = (256, 256, 256, 256)
        dim_reduced = 256

        self.roi_heads.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)       
        self.roi_heads.mask_head = MaskRCNNHeads(out_channels, layers, 1)
        self.roi_heads.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        #Image Transformer
        self.transformer = ImageTransforms(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])

    def load_from_pretrained(self,num_classes):
        fn = Path('./') / "weights/maskrcnn.pt"
        download_file("https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth", fn)

        with open(fn, "rb") as f:
            state_dict = fake_torch_load(f.read())
        loaded_keys = []
        pretrained_msd = list(state_dict.items())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        i = 0
        for k, v in state_dict.items():
            if i in skip_list:
                i+=1
                continue
            try:
                get_child(self, k).assign(v.numpy()).realize()
                loaded_keys.append(k)
            except Exception as e: print(k,e)
            i+=1
        return loaded_keys
        
    def __call__(self, image, target=None):
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

class FastRCNNConvFCHead:
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        conv_layers: List[int],
        fc_layers: List[int],
        norm_layer = None,
    ):
        in_channels, in_height, in_width = input_size

        self.blocks = []
        previous_channels = in_channels
        for current_channels in conv_layers:
            self.blocks.append(F.Conv2dNormActivation(previous_channels, current_channels))
            previous_channels = current_channels
        self.blocks.append(F.Flatten())
        previous_channels = previous_channels * in_height * in_width
        for current_channels in fc_layers:
            self.blocks.append(nn.Linear(previous_channels, current_channels))
            self.blocks.append(F.ReLU(inplace=True))
            previous_channels = current_channels

        for layer in self.blocks:
            if isinstance(layer, nn.Conv2d):
                F.kaiming_normal_init(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    layer.bias.zeros

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x

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
 
class MaskRCNNHeads:
    _version = 2

    def __init__(self, in_channels, layers, dilation, norm_layer = None):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        blocks = []
        next_feature = in_channels
        
        self.mask_fcn1 = F.Conv2dNormActivation(256, layers[0], dilation=dilation, stride=1, use_gn=False)
        self.mask_fcn2 = F.Conv2dNormActivation(layers[0], layers[1], dilation=dilation, stride=1, use_gn=False)
        self.mask_fcn3 = F.Conv2dNormActivation(layers[1], layers[2], dilation=dilation, stride=1, use_gn=False)
        self.mask_fcn4 = F.Conv2dNormActivation(layers[2], layers[3], dilation=dilation, stride=1, use_gn=False)

        self.blocks = [self.mask_fcn1, self.mask_fcn2, self.mask_fcn3, self.mask_fcn4]

        for layer in self.blocks:
            if isinstance(layer, nn.Conv2d):
                F.kaiming_normal_init(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    layer.bias.zeros

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            num_blocks = len(self.blocks)
            for i in range(num_blocks):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}mask_fcn{i+1}.{type}"
                    new_key = f"{prefix}{i}.0.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

        self.conv5_mask = nn.ConvTranspose2d(next_feature, dim_reduced, kernel_size=2, stride=2, padding=0)
        self.relu5 = F.ReLU(inplace=True)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, kernel_size=1, stride=1, padding=0)


        for layer in self.mask_layers:
            if isinstance(layer, nn.Conv2d):
                layer = F.kaiming_normal_init(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        for layer in self.mask_layers:
            x = layer(x)

        x = self.conv5_mask(x)
        x = self.relu5(x)
        x = self.mask_fcn_logits(x)

        return x

class ResBackbone:
    def __init__(self, backbone, pretrained):
        super().__init__()

        def forward(self,x):
            out = self.bn1(self.conv1(x)).relu()
            out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
            out1 = out.sequential(self.layer1)
            out2 = out1.sequential(self.layer2)
            out3 = out2.sequential(self.layer3)
            out4 = out3.sequential(self.layer4)
            return [out1, out2, out3, out4]   
        
        self.out_channels = 256

        body = OrderedDict()
        for attr, value in backbone.__dict__.items():
            if attr != 'fc':
                body[attr] = value

        backbone.__dict__ = body

        backbone.forward = forward.__get__(backbone, ResNet)

        self.body = backbone

        in_channels = 256
        in_channels_list = [
        in_channels,
        in_channels * 2,
        in_channels * 4,
        in_channels * 8,
        ]
        self.fpn = FPN(in_channels_list, self.out_channels)

    def __call__(self, x):
        x = self.body(x)
        return self.fpn(x)

def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True):
    if pretrained:
        backbone_pretrained = False
    resnet = resnet = ResNet(50, num_classes=num_classes)
    backbone = ResBackbone( resnet, pretrained_backbone)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )

    model = MaskRCNN(backbone, num_classes, box_head)

    if pretrained:
        model.load_from_pretrained(num_classes)
    return model