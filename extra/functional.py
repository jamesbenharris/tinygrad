import math
import numpy as np
from typing import Tuple,List,Optional,Union,Callable
import tinygrad.nn as nn
#Need to validate
class ReLU:
    def __init__(self, inplace=False):
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            np.maximum(input, 0, out=input)
            return input
        else:
            return np.maximum(0, input)

#Need to validate
class Flatten:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x.flatten()

def zeros_(tensor):
    tensor.fill(0.0)
#Need to validate 
def Conv2dNormActivation(
  in_channels,
  out_channels,
  dilation=1,
  stride=1,
  use_gn=False,
):
  conv = nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=stride,
    padding=dilation,
    dilation=dilation,
    bias=False if use_gn else True
  )
  return conv


#Need to validate          
def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    batch_size, num_channels, input_height, input_width = input.shape
    num_rois = boxes.shape[0]
    output_height, output_width = output_size

    rois = boxes * spatial_scale

    if aligned:
        rois[:, [0, 2]] -= 0.5
        rois[:, [1, 3]] -= 0.5

    rois[:, [0, 2]] = np.clip(rois[:, [0, 2]], 0, input_width - 1)
    rois[:, [1, 3]] = np.clip(rois[:, [1, 3]], 0, input_height - 1)

    roi_heights = np.maximum(rois[:, 3] - rois[:, 1] + 1, 1)
    roi_widths = np.maximum(rois[:, 2] - rois[:, 0] + 1, 1)

    bin_height = roi_heights / output_height
    bin_width = roi_widths / output_width

    if sampling_ratio > 0:
        sampling_points = sampling_ratio
    else:
        sampling_points = np.ceil(np.maximum(roi_heights / output_height, roi_widths / output_width)).astype(int)

    pooled_rois = np.zeros((num_rois, num_channels, output_height, output_width))

    for i in range(num_rois):
        for c in range(num_channels):
            y_indices = np.linspace(rois[i, 1], rois[i, 3], output_height * sampling_points)
            x_indices = np.linspace(rois[i, 0], rois[i, 2], output_width * sampling_points)

            y_grid, x_grid = np.meshgrid(y_indices, x_indices)

            y0 = np.floor(y_grid).astype(int)
            y1 = y0 + 1
            x0 = np.floor(x_grid).astype(int)
            x1 = x0 + 1

            y0 = np.clip(y0, 0, input_height - 1)
            y1 = np.clip(y1, 0, input_height - 1)
            x0 = np.clip(x0, 0, input_width - 1)
            x1 = np.clip(x1, 0, input_width - 1)

            Q11 = input[0, c, y0, x0]
            Q12 = input[0, c, y0, x1]
            Q21 = input[0, c, y1, x0]
            Q22 = input[0, c, y1, x1]

            L = (y1 - y_grid) * (x1 - x_grid)
            R = (y1 - y_grid) * (x_grid - x0)
            T = (y_grid - y0) * (x1 - x_grid)
            B = (y_grid - y0) * (x_grid - x0)

            sampled_values = L * Q11 + R * Q12 + T * Q21 + B * Q22
            pooled_values = np.mean(sampled_values.reshape(output_height, output_width, sampling_points, sampling_points), axis=(2, 3))

            pooled_rois[i, c] = pooled_values

    return pooled_rois

#Need to validate
def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif isinstance(param, (int, float)):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return np.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

class LastLevelMaxPool:
  def __call__(self, x):
    return [Tensor.max_pool2d(x, 1, 2)]

#Need to validate
def kaiming_normal_init(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    fan = np.prod(tensor.shape[:-1]) if mode == 'fan_out' else np.prod(tensor.shape[1:])
    gain = calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    return np.random.normal(0, std, size=tensor.shape)

def relu(input, inplace=False):
    if inplace:
        np.maximum(input, 0, out=input)
        return input
    else:
        return np.maximum(0, input)

#Need to validate
def cross_entropy(p, q):
    return -np.sum(p * np.log(q))

#Need to validate
def smooth_l1_loss(predictions, targets, beta=1.0):
    loss = np.zeros_like(predictions)
    diff = predictions - targets
    mask = np.abs(diff) < beta
    loss += mask * (0.5 * diff ** 2 / beta)
    loss += (~mask) * (np.abs(diff) - 0.5 * beta)
    return np.mean(loss)

#Need to validate
def binary_cross_entropy_with_logits(input, target, weight=None, pos_weight=None, reduction='mean'):
    p = np.array([1.0])
    sig_x = 1 / (1 + np.exp(-input))
    log_sig_x = np.log(sig_x)
    sub_1_x = p - sig_x
    sub_1_y = p - target
    log_1_x = np.log(sub_1_x)
    
    if pos_weight is None:
        output = -((target * log_sig_x) + (sub_1_y * log_1_x))
    else:
        output = -((target * log_sig_x * pos_weight) + (sub_1_y * log_1_x))
    
    if reduction == 'mean':
        return np.mean(output)
    elif reduction == 'sum':
        return np.sum(output)
    else:
        return outputs

#Need to validate
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

#Need to validate
def pad(input, pad):
    if isinstance(pad, int):
        pad = (pad, pad)  # Convert single integer to tuple for symmetric padding
    elif isinstance(pad, tuple):
        if len(pad) != len(input.shape):
            raise ValueError("The length of 'pad' tuple must match the input's number of dimensions.")
    else:
        raise TypeError("Invalid type for 'pad'. Expected int or tuple.")

    # Calculate the padding for each dimension
    pad_width = [(p, p) for p in pad]

    # Perform padding using numpy.pad
    padded_input = np.pad(input, pad_width, mode='constant')

    return padded_input

#Need to validate
def interpolate(input, size=None, scale_factor=None, mode='bilinear'):
    if size is None and scale_factor is None:
        raise ValueError("Either 'size' or 'scale_factor' must be specified.")

    if scale_factor is not None:
        if isinstance(scale_factor, float):
            scale_factor = (scale_factor, scale_factor)
        output_size = tuple(int(dim * factor) for dim, factor in zip(input.shape[2:], scale_factor))
    else:
        output_size = tuple(size)

    if mode == 'nearest':
        interpolation = np.round
    elif mode == 'bilinear':
        interpolation = np.interp
    else:
        raise ValueError("Invalid interpolation mode. Only 'nearest' and 'bilinear' are supported.")

    output = np.zeros((input.shape[0], input.shape[1], output_size[0], output_size[1]), dtype=input.dtype)

    for batch_idx in range(input.shape[0]):
        for channel_idx in range(input.shape[1]):
            for out_row in range(output_size[0]):
                for out_col in range(output_size[1]):
                    in_row = (out_row + 0.5) * input.shape[2] / output_size[0] - 0.5
                    in_col = (out_col + 0.5) * input.shape[3] / output_size[1] - 0.5
                    in_row = np.clip(in_row, 0, input.shape[2] - 1)
                    in_col = np.clip(in_col, 0, input.shape[3] - 1)
                    in_row_low, in_row_high = int(np.floor(in_row)), min(int(np.floor(in_row)) + 1, input.shape[2] - 1)
                    in_col_low, in_col_high = int(np.floor(in_col)), min(int(np.floor(in_col)) + 1, input.shape[3] - 1)
                    weight_row_high = in_row - in_row_low
                    weight_row_low = 1 - weight_row_high
                    weight_col_high = in_col - in_col_low
                    weight_col_low = 1 - weight_col_high
                    output[batch_idx, channel_idx, out_row, out_col] = (
                        weight_row_low * (weight_col_low * input[batch_idx, channel_idx, in_row_low, in_col_low] +
                                          weight_col_high * input[batch_idx, channel_idx, in_row_low, in_col_high]) +
                        weight_row_high * (weight_col_low * input[batch_idx, channel_idx, in_row_high, in_col_low] +
                                           weight_col_high * input[batch_idx, channel_idx, in_row_high, in_col_high])
                    )

    return output
