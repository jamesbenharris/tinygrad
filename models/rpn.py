from tinygrad.tensor import Tensor
import tinygrad.nn as nn
import extra.functional as F

from extra.boxops import BoxCoder,process_box,nms
from extra.utils import Matcher,BalancedPositiveNegativeSampler

#Source https://github.com/Okery/PyTorch-Simple-MaskRCNN

class RPNHead:
  def __init__(self, in_channels, num_anchors):
    self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
    self.cls_logits = nn.Conv2d(256, num_anchors, kernel_size=1)
    self.bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=1)

  def __call__(self, x):
    logits = []
    bbox_reg = []
    for feature in x:
        t = Tensor.relu(self.conv(feature))
        logits.append(self.cls_logits(t))
        bbox_reg.append(self.bbox_pred(t))
    return logits, bbox_reg
    

class RegionProposalNetwork:
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        
        self.anchor_generator = anchor_generator
        self.head = head
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1
                
    def create_proposal(self, anchor, objectness, pred_bbox_delta, image_shape):
        if self.training:
            pre_nms_top_n = self._pre_nms_top_n['training']
            post_nms_top_n = self._post_nms_top_n['training']
        else:
            pre_nms_top_n = self._pre_nms_top_n['testing']
            post_nms_top_n = self._post_nms_top_n['testing']
            
        pre_nms_top_n = min(objectness.shape[0], pre_nms_top_n)
        top_n_idx = objectness.argsort(descending=True)[:pre_nms_top_n]
        score = objectness[top_n_idx]
        proposal = self.box_coder.decode(pred_bbox_delta[top_n_idx], anchor[top_n_idx])
        
        proposal, score = process_box(proposal, score, image_shape, self.min_size)
        keep = nms(proposal, score, self.nms_thresh)[:post_nms_top_n] 
        proposal = proposal[keep]
        return proposal
    
    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        iou = box_iou(gt_box, anchor)
        label, matched_idx = self.proposal_matcher(iou)
        
        pos_idx, neg_idx = self.fg_bg_sampler(label)
        idx = np.concatenate((pos_idx, neg_idx))
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])
        
        objectness_loss = binary_cross_entropy_with_logits(objectness[idx], label[idx])
        box_loss = l1_loss(pred_bbox_delta[pos_idx], regression_target) / idx.size

        return objectness_loss, box_loss
        
    def forward(self, feature, image_shape, target=None):
        if target is not None:
            gt_box = target['boxes']
        anchor = self.anchor_generator(feature, image_shape)
        
        objectness, pred_bbox_delta = self.head(feature)
        objectness = objectness.permute(0, 2, 3, 1).flatten()
        pred_bbox_delta = pred_bbox_delta.permute(0, 2, 3, 1).reshape(-1, 4)

        proposal = self.create_proposal(anchor, objectness.detach(), pred_bbox_delta.detach(), image_shape)
        if self.training:
            objectness_loss, box_loss = self.compute_loss(objectness, pred_bbox_delta, gt_box, anchor)
            return proposal, dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)
        
        return proposal, {}