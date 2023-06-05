import math
import numpy as np
from typing import Tuple,List,Optional,Union,Callable
import tinygrad.nn as nn
from PIL import Image
from tinygrad.tensor import Tensor

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

def constant(tensor,value):
    return np.full(tensor.data.shape,value)

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError("align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear")
    else:
        if align_corners is None:
            align_corners = False

    dim = len(input.shape) - 2  # Number of spatial dimensions.
    ndim = len(input.shape)   # Number of dimensions.
    
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(f"Input and output must have the same number of spatial dimensions, but got input with spatial dimensions of {list(input.shape[2:])} and output size of {size}. Please provide input tensor in (N, C, d1, d2, ..., dK) format and output size in (o1, o2, ..., oK) format.")
            output_size = size
        else:
            output_size = [size for _ in range(dim)]
    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(f"Input and scale_factor must have the same number of spatial dimensions, but got input with spatial dimensions of {list(input.shape[2:])} and scale_factor of shape {scale_factor}. Please provide input tensor in (N, C, d1, d2, ..., dK) format and scale_factor in (s1, s2, ..., sK) format.")
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
    else:
        raise ValueError("either size or scale_factor should be defined")

    if recompute_scale_factor is not None and recompute_scale_factor and size is not None:
        raise ValueError("recompute_scale_factor is not meaningful with an explicit size.")

    if mode == "area" and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        assert scale_factors is not None
        output_size = [
            math.floor(input.shape[i + 2] * scale_factors[i])
            for i in range(dim)
        ]
        scale_factors = None

    if antialias and not (mode in ("bilinear", "bicubic") and ndim == 4):
        raise ValueError("Anti-alias option is only supported for bilinear and bicubic modes")

    if ndim == 3 and mode == "nearest":
        return upsample_nearest1d(input, output_size, scale_factors)
    if ndim == 4 and mode == "nearest":
        return upsample_nearest2d(input, output_size, scale_factors)
    if ndim == 5 and mode == "nearest":
        return upsample_nearest3d(input, output_size, scale_factors)

    if ndim == 3 and mode == "nearest-exact":
        return upsample_nearest_exact1d(input, output_size, scale_factors)
    if ndim == 4 and mode == "nearest-exact":
        return upsample_nearest_exact2d(input, output_size, scale_factors)
    if ndim == 5 and mode == "nearest-exact":
        return upsample_nearest_exact3d(input, output_size, scale_factors)

    if ndim == 3 and mode == "area":
        return upsample_area1d(input, output_size, scale_factors)
    if ndim == 4 and mode == "area":
        return upsample_area2d(input, output_size, scale_factors)
    if ndim == 5 and mode == "area":
        return upsample_area3d(input, output_size, scale_factors)

    if ndim == 3 and mode == "linear":
        return upsample_linear1d(input, output_size, scale_factors, align_corners)
    if ndim == 4 and mode == "bilinear":
        return upsample_bilinear2d(input, output_size, scale_factors, align_corners, antialias)
    if ndim == 5 and mode == "trilinear":
        return upsample_trilinear3d(input, output_size, scale_factors, align_corners, antialias)

    if ndim == 4 and mode == "bicubic":
        return upsample_bicubic2d(input, output_size, scale_factors, align_corners, antialias)

    raise ValueError(f"Input Error: Only 3D, 4D, and 5D input Tensors supported (got {ndim}D) for the modes: nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact")

def upsample_nearest1d(input, output_size=None, scale_factors=None):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2]]

    return np.repeat(input, np.round(scale_factors[0]).astype(int), axis=2)

def upsample_nearest2d(input, output_size=None, scale_factors=None):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2], output_size[1] / input.shape[3]]

    return np.repeat(np.repeat(input, np.round(scale_factors[0]).astype(int), axis=2), np.round(scale_factors[1]).astype(int), axis=3)

def upsample_nearest3d(input, output_size=None, scale_factors=None):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2], output_size[1] / input.shape[3], output_size[2] / input.shape[4]]

    return np.repeat(np.repeat(np.repeat(input, np.round(scale_factors[0]).astype(int), axis=2), np.round(scale_factors[1]).astype(int), axis=3), np.round(scale_factors[2]).astype(int), axis=4)

def upsample_nearest_exact1d(input, output_size=None, scale_factors=None):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2]]

    if scale_factors[0] == 1.0:
        return input

    return np.interp(
        np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0])),
        np.linspace(0, 1, num=input.shape[2]),
        input,
        axis=2
    )

def upsample_nearest_exact2d(input, output_size=None, scale_factors=None):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2], output_size[1] / input.shape[3]]

    if scale_factors[0] == 1.0 and scale_factors[1] == 1.0:
        return input

    x = np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0]))
    y = np.linspace(0, 1, num=int(input.shape[3] * scale_factors[1]))

    return np.interp(
        y[:, np.newaxis],
        np.linspace(0, 1, num=input.shape[3]),
        np.interp(
            x,
            np.linspace(0, 1, num=input.shape[2]),
            input,
            axis=2
        ),
        axis=1
    )

def upsample_nearest_exact3d(input, output_size=None, scale_factors=None):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2], output_size[1] / input.shape[3], output_size[2] / input.shape[4]]

    if scale_factors[0] == 1.0 and scale_factors[1] == 1.0 and scale_factors[2] == 1.0:
        return input

    x = np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0]))
    y = np.linspace(0, 1, num=int(input.shape[3] * scale_factors[1]))
    z = np.linspace(0, 1, num=int(input.shape[4] * scale_factors[2]))

    return np.interp(
        z[:, np.newaxis, np.newaxis],
        np.linspace(0, 1, num=input.shape[4]),
        np.interp(
            y[:, np.newaxis],
            np.linspace(0, 1, num=input.shape[3]),
            np.interp(
                x,
                np.linspace(0, 1, num=input.shape[2]),
                input,
                axis=2
            ),
            axis=1
        ),
        axis=0
    )

def upsample_area1d(input, output_size=None, scale_factors=None):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2]]

    return np.repeat(input, np.round(scale_factors[0]).astype(int), axis=2)

def upsample_area2d(input, output_size=None, scale_factors=None):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2], output_size[1] / input.shape[3]]

    return np.repeat(np.repeat(input, np.round(scale_factors[0]).astype(int), axis=2), np.round(scale_factors[1]).astype(int), axis=3)

def upsample_area3d(input, output_size=None, scale_factors=None):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2], output_size[1] / input.shape[3], output_size[2] / input.shape[4]]

    return np.repeat(np.repeat(np.repeat(input, np.round(scale_factors[0]).astype(int), axis=2), np.round(scale_factors[1]).astype(int), axis=3), np.round(scale_factors[2]).astype(int), axis=4)

def upsample_linear1d(input, output_size=None, scale_factors=None, align_corners=False):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2]]

    if align_corners:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=input.shape[2]),
            input,
            axis=2
        )
    else:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0])),
            input,
            axis=2
        )

def upsample_bilinear2d(input, output_size=None, scale_factors=None, align_corners=False, antialias=False):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2], output_size[1] / input.shape[3]]

    if align_corners and antialias:
        raise ValueError("align_corners and antialias options are incompatible")

    if align_corners:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=input.shape[2]),
            np.interp(
                np.linspace(0, 1, num=output_size[1]),
                np.linspace(0, 1, num=input.shape[3]),
                input,
                axis=3
            ),
            axis=2
        )
    elif antialias:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0])),
            np.interp(
                np.linspace(0, 1, num=output_size[1]),
                np.linspace(0, 1, num=int(input.shape[3] * scale_factors[1])),
                input,
                axis=3
            ),
            axis=2
        )
    else:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0])),
            np.interp(
                np.linspace(0, 1, num=output_size[1]),
                np.linspace(0, 1, num=int(input.shape[3] * scale_factors[1])),
                input,
                axis=3
            ),
            axis=2
        )

def upsample_trilinear3d(input, output_size=None, scale_factors=None, align_corners=False, antialias=False):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2], output_size[1] / input.shape[3], output_size[2] / input.shape[4]]

    if align_corners and antialias:
        raise ValueError("align_corners and antialias options are incompatible")

    if align_corners:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=input.shape[2]),
            np.interp(
                np.linspace(0, 1, num=output_size[1]),
                np.linspace(0, 1, num=input.shape[3]),
                np.interp(
                    np.linspace(0, 1, num=output_size[2]),
                    np.linspace(0, 1, num=input.shape[4]),
                    input,
                    axis=4
                ),
                axis=3
            ),
            axis=2
        )
    elif antialias:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0])),
            np.interp(
                np.linspace(0, 1, num=output_size[1]),
                np.linspace(0, 1, num=int(input.shape[3] * scale_factors[1])),
                np.interp(
                    np.linspace(0, 1, num=output_size[2]),
                    np.linspace(0, 1, num=int(input.shape[4] * scale_factors[2])),
                    input,
                    axis=4
                ),
                axis=3
            ),
            axis=2
        )
    else:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0])),
            np.interp(
                np.linspace(0, 1, num=output_size[1]),
                np.linspace(0, 1, num=int(input.shape[3] * scale_factors[1])),
                np.interp(
                    np.linspace(0, 1, num=output_size[2]),
                    np.linspace(0, 1, num=int(input.shape[4] * scale_factors[2])),
                    input,
                    axis=4
                ),
                axis=3
            ),
            axis=2
        )

def upsample_bicubic2d(input, output_size=None, scale_factors=None, align_corners=False, antialias=False):
    if scale_factors is None:
        if output_size is None:
            raise ValueError("either size or scale_factor should be defined")
        scale_factors = [output_size[0] / input.shape[2], output_size[1] / input.shape[3]]

    if align_corners and antialias:
        raise ValueError("align_corners and antialias options are incompatible")

    if align_corners:
        raise ValueError("align_corners option is not supported for bicubic interpolation")

    if antialias:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0])),
            np.interp(
                np.linspace(0, 1, num=output_size[1]),
                np.linspace(0, 1, num=int(input.shape[3] * scale_factors[1])),
                input,
                axis=3
            ),
            axis=2
        )
    else:
        return np.interp(
            np.linspace(0, 1, num=output_size[0]),
            np.linspace(0, 1, num=int(input.shape[2] * scale_factors[0])),
            np.interp(
                np.linspace(0, 1, num=output_size[1]),
                np.linspace(0, 1, num=int(input.shape[3] * scale_factors[1])),
                input,
                axis=3
            ),
            axis=2
        )
