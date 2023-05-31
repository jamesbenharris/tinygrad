import math
from tinygrad.tensor import Tensor

class BoxCoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_box, proposal):
        width = proposal[:, 2] - proposal[:, 0]
        height = proposal[:, 3] - proposal[:, 1]
        ctr_x = proposal[:, 0] + 0.5 * width
        ctr_y = proposal[:, 1] + 0.5 * height

        gt_width = reference_box[:, 2] - reference_box[:, 0]
        gt_height = reference_box[:, 3] - reference_box[:, 1]
        gt_ctr_x = reference_box[:, 0] + 0.5 * gt_width
        gt_ctr_y = reference_box[:, 1] + 0.5 * gt_height

        dx = self.weights[0] * (gt_ctr_x - ctr_x) / width
        dy = self.weights[1] * (gt_ctr_y - ctr_y) / height
        dw = self.weights[2] * Tensor.log(gt_width / width)
        dh = self.weights[3] * Tensor.log(gt_height / height)

        delta = Tensor.stack((dx, dy, dw, dh), dim=1)
        return delta

    def decode(self, delta, box):
        dx = delta[:, 0] / self.weights[0]
        dy = delta[:, 1] / self.weights[1]
        dw = delta[:, 2] / self.weights[2]
        dh = delta[:, 3] / self.weights[3]

        dw = Tensor.clamp(dw, max=self.bbox_xform_clip)
        dh = Tensor.clamp(dh, max=self.bbox_xform_clip)

        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        ctr_x = box[:, 0] + 0.5 * width
        ctr_y = box[:, 1] + 0.5 * height

        pred_ctr_x = dx * width + ctr_x
        pred_ctr_y = dy * height + ctr_y
        pred_w = Tensor.exp(dw) * width
        pred_h = Tensor.exp(dh) * height

        xmin = pred_ctr_x - 0.5 * pred_w
        ymin = pred_ctr_y - 0.5 * pred_h
        xmax = pred_ctr_x + 0.5 * pred_w
        ymax = pred_ctr_y + 0.5 * pred_h

        target = Tensor.stack((xmin, ymin, xmax, ymax), dim=1)
        return target

def box_iou(box_a, box_b):
    lt = Tensor.max(box_a[:, None, :2], box_b[:, :2])
    rb = Tensor.min(box_a[:, None, 2:], box_b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = Tensor.prod(box_a[:, 2:] - box_a[:, :2], 1)
    area_b = Tensor.prod(box_b[:, 2:] - box_b[:, :2], 1)
    
    return inter / (area_a[:, None] + area_b - inter)

def process_box(box, score, image_shape, min_size):
    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, image_shape[1]) 
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, image_shape[0]) 

    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    keep = Tensor.where((w >= min_size) & (h >= min_size))[0]
    box, score = box[keep], score[keep]
    return box, score

def nms(boxes, scores, thresh=0.5):
  x1, y1, x2, y2 = np.rollaxis(boxes, 1)
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  to_process, keep = scores.argsort()[::-1], []
  while to_process.size > 0:
    cur, to_process = to_process[0], to_process[1:]
    keep.append(cur)
    inter_x1 = np.maximum(x1[cur], x1[to_process])
    inter_y1 = np.maximum(y1[cur], y1[to_process])
    inter_x2 = np.minimum(x2[cur], x2[to_process])
    inter_y2 = np.minimum(y2[cur], y2[to_process])
    inter_area = np.maximum(0, inter_x2 - inter_x1 + 1) * np.maximum(0, inter_y2 - inter_y1 + 1)
    iou = inter_area / (areas[cur] + areas[to_process] - inter_area)
    to_process = to_process[np.where(iou <= thresh)[0]]
  return keep
