import pickle
import numpy as np
from tqdm import tqdm
import tempfile, platform
from collections import defaultdict
from tinygrad.helpers import prod, getenv, DEBUG
from tinygrad.ops import GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad.lazy import LazyNumpyArray, Device
from tinygrad.shape.shapetracker import strides_for_shape
import extra.functional as F

OSX = platform.system() == "Darwin"

def fetch(url):
  if url.startswith("/"):
    with open(url, "rb") as f:
      return f.read()
  import os, hashlib, tempfile
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
  download_file(url, fp, skip_if_exists=not getenv("NOCACHE"))
  with open(fp, "rb") as f:
    return f.read()

def download_file(url, fp, skip_if_exists=True):
  import requests, os, pathlib
  if skip_if_exists and os.path.isfile(fp) and os.stat(fp).st_size > 0:
    return
  r = requests.get(url, stream=True)
  assert r.status_code == 200
  progress_bar = tqdm(total=int(r.headers.get('content-length', 0)), unit='B', unit_scale=True, desc=url)
  with tempfile.NamedTemporaryFile(dir=pathlib.Path(fp).parent, delete=False) as f:
    for chunk in r.iter_content(chunk_size=16384):
      progress_bar.update(f.write(chunk))
    f.close()
    os.rename(f.name, fp)


def my_unpickle(fb0):
  key_prelookup = defaultdict(list)
  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    ident, storage_type, obj_key, location, obj_size = storage[0:5]
    assert ident == 'storage'
    assert prod(size) <= (obj_size - storage_offset)

    if storage_type not in [np.float16, np.float32]:
      if DEBUG: print(f"unsupported type {storage_type} on {obj_key} with shape {size}")
      ret = None
    else:
      ret = Tensor(LazyNumpyArray(lambda lst: np.zeros(lst.shape, dtype=lst.dtype), tuple(size), storage_type))
    key_prelookup[obj_key].append((storage_type, obj_size, ret, size, stride, storage_offset))
    return ret

  def _rebuild_parameter(*args):
    #print(args)
    pass

  class Dummy: pass

  class MyPickle(pickle.Unpickler):
    def find_class(self, module, name):
      #print(module, name)
      if name == 'FloatStorage': return np.float32
      if name == 'LongStorage': return np.int64
      if name == 'IntStorage': return np.int32
      if name == 'HalfStorage': return np.float16
      if module == "torch._utils":
        if name == "_rebuild_tensor_v2": return _rebuild_tensor_v2
        if name == "_rebuild_parameter": return _rebuild_parameter
      else:
        if module.startswith('pytorch_lightning'): return Dummy
        try:
          return super().find_class(module, name)
        except Exception:
          return Dummy

    def persistent_load(self, pid):
      return pid

  return MyPickle(fb0).load(), key_prelookup

def load_single_weight(t:Tensor, myfile, shape, strides, dtype, storage_offset, mmap_allowed=False):
  bytes_size = np.dtype(dtype).itemsize
  if t is None:
    myfile.seek(prod(shape) * bytes_size, 1)
    return

  bytes_offset = 0
  if storage_offset is not None:
    bytes_offset = storage_offset * bytes_size
    myfile.seek(bytes_offset)

  assert t.shape == shape or shape == tuple(), f"shape mismatch {t.shape} != {shape}"
  assert t.dtype.np == dtype and t.dtype.itemsize == bytes_size
  if any(s != 1 and st1 != st2 for s, st1, st2 in zip(shape, strides_for_shape(shape), strides)):
    # slow path
    buffer_size = sum(strides[i]*t.dtype.itemsize * (shape[i] - 1) for i in range(len(shape)))
    buffer_size += t.dtype.itemsize
    np_array = np.frombuffer(myfile.read(buffer_size), t.dtype.np)

    np_array = np.lib.stride_tricks.as_strided(
      np_array, shape=shape, strides=[i*t.dtype.itemsize for i in strides])

    lna = t.lazydata.op.arg
    lna.fxn = lambda _: np_array
    t.realize()
    return

  # ["METAL", "CLANG", "LLVM"] support readinto for more speed
  # ["GPU", "CUDA"] use _mmap since they have to copy in to the GPU anyway
  # this needs real APIs
  if t.device in ["METAL", "CLANG", "LLVM"]:
    del t.lazydata.op
    t.lazydata.realized = Device[t.lazydata.device].buffer(prod(t.shape), dtype=t.dtype)
    myfile.readinto(t.lazydata.realized._buffer())
  else:
    def _mmap(lna):
      assert myfile._compress_type == 0, "compressed data can't be mmaped"
      return np.memmap(myfile._fileobj._file, dtype=lna.dtype, mode='r', offset=myfile._orig_compress_start + bytes_offset, shape=lna.shape)
    def _read(lna):
      ret = np.empty(lna.shape, dtype=lna.dtype)
      myfile.readinto(ret.data)
      return ret
    if mmap_allowed and not OSX and t.device in ["GPU", "CUDA"]: t.lazydata.op.arg.fxn = _mmap
    else: t.lazydata.op.arg.fxn = _read
    t.realize()

def fake_torch_load_zipped(fb0, load_weights=True, multithreaded=True):
  if Device.DEFAULT in ["TORCH", "GPU", "CUDA"]: multithreaded = False  # multithreaded doesn't work with CUDA or TORCH. for GPU it's a wash with _mmap

  import zipfile
  with zipfile.ZipFile(fb0, 'r') as myzip:
    base_name = myzip.namelist()[0].split('/', 1)[0]
    with myzip.open(f'{base_name}/data.pkl') as myfile:
      ret = my_unpickle(myfile)
    if load_weights:
      def load_weight(k, vv):
        with myzip.open(f'{base_name}/data/{k}') as myfile:
          for v in vv:
            load_single_weight(v[2], myfile, v[3], v[4], v[0], v[5], mmap_allowed=True)
      if multithreaded:
        import concurrent.futures
        # 2 seems fastest
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
          futures = {executor.submit(load_weight, k, v):k for k,v in ret[1].items()}
          for future in (t:=tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
            if future.exception() is not None: raise future.exception()
            k = futures[future]
            t.set_description(f"loading {k} ram used: {GlobalCounters.mem_used/1e9:5.2f} GB")
      else:
        for k,v in (t := tqdm(ret[1].items())):
          t.set_description(f"loading {k} ram used: {GlobalCounters.mem_used/1e9:5.2f} GB")
          load_weight(k,v)
  return ret[0]

def fake_torch_load(b0):
  import io
  import struct

  # convert it to a file
  fb0 = io.BytesIO(b0)

  if b0[0:2] == b"\x50\x4b":
    return fake_torch_load_zipped(fb0)

  # skip three junk pickles
  pickle.load(fb0)
  pickle.load(fb0)
  pickle.load(fb0)

  ret, key_prelookup = my_unpickle(fb0)

  # create key_lookup
  key_lookup = pickle.load(fb0)
  key_real = [None] * len(key_lookup)
  for k,v in key_prelookup.items():
    assert len(v) == 1
    key_real[key_lookup.index(k)] = v[0]

  # read in the actual data
  for storage_type, obj_size, tensor, np_shape, np_strides, storage_offset in key_real:
    ll = struct.unpack("Q", fb0.read(8))[0]
    assert ll == obj_size, f"size mismatch {ll} != {obj_size}"
    assert storage_offset == 0, "not implemented"
    load_single_weight(tensor, fb0, np_shape, np_strides, storage_type, None)

  return ret

def get_child(parent, key):
  obj = parent
  for k in key.split('.'):
    if k.isnumeric():
      obj = obj[int(k)]
    elif isinstance(obj, dict):
      obj = obj[k]
    else:
      obj = getattr(obj, k)
  return obj

def roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    if torch.__version__ >= "1.5.0":
        return Tensor.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, False)
    else:
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)


class AnchorGenerator:
    def __init__(self, sizes, ratios):
        self.sizes = sizes
        self.ratios = ratios
        
        self.cell_anchor = None
        self._cache = {}
        
    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return 
        sizes = Tensor.tensor(self.sizes, dtype=dtype, device=device)
        ratios = Tensor.tensor(self.ratios, dtype=dtype, device=device)

        h_ratios = Tensor.sqrt(ratios)
        w_ratios = 1 / h_ratios

        hs = (sizes[:, None] * h_ratios[None, :]).view(-1)
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1)

        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        
    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_x = Tensor.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1]
        shift_y = Tensor.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]

        y, x = Tensor.meshgrid(shift_y, shift_x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        shift = Tensor.stack((x, y, x, y), dim=1).reshape(-1, 1, 4)

        anchor = (shift + self.cell_anchor).reshape(-1, 4)
        return anchor
        
    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride
        if key in self._cache:
            return self._cache[key]
        anchor = self.grid_anchor(grid_size, stride)
        
        if len(self._cache) >= 3:
            self._cache.clear()
        self._cache[key] = anchor
        return anchor

    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:])
        stride = tuple(int(i / g) for i, g in zip(image_size, grid_size))
        
        self.set_cell_anchor(dtype, device)
        
        anchor = self.cached_grid_anchor(grid_size, stride)
        return anchor

class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
        
        value, matched_idx = iou.max(dim=0)
        label = Tensor.full((iou.shape[1],), -1, dtype=Tensor.float, device=iou.device) 
        
        label[value >= self.high_threshold] = 1
        label[value < self.low_threshold] = 0
        
        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0]
            gt_pred_pairs = Tensor.where(iou == highest_quality[:, None])[1]
            label[gt_pred_pairs] = 1

        return label, matched_idx
    

class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = Tensor.where(label == 1)[0]
        negative = Tensor.where(label == 0)[0]
        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)
        
        pos_perm = Tensor.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = Tensor.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx

class ImageTransforms:
    def __init__(self, min_size, max_size, image_mean, image_std):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        
    def __call__(self, image, target):
        image = self.normalize(image)
        image, target = self.resize(image, target)
        image = self.batched_image(image)
        
        return image, target

    def normalize(self, image):
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        dtype, device = image.dtype, image.device
        mean = Tensor.tensor(self.image_mean, dtype=dtype, device=device)
        std = Tensor.tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        ori_image_shape = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        
        scale_factor = min(self.min_size / min_size, self.max_size / max_size)
        size = [round(s * scale_factor) for s in ori_image_shape]
        image = F.interpolate(image[None], size=size, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target
        
        box = target['boxes']
        box[:, [0, 2]] = box[:, [0, 2]] * image.shape[-1] / ori_image_shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] * image.shape[-2] / ori_image_shape[0]
        target['boxes'] = box
        
        if 'masks' in target:
            mask = target['masks']
            mask = F.interpolate(mask[None].float(), size=size)[0].byte()
            target['masks'] = mask
            
        return image, target
    
    def batched_image(self, image, stride=32):
        size = image.shape[-2:]
        max_size = tuple(math.ceil(s / stride) * stride for s in size)

        batch_shape = (image.shape[-3],) + max_size
        batched_img = image.new_full(batch_shape, 0)
        batched_img[:, :image.shape[-2], :image.shape[-1]] = image

        return batched_img[None]
    
    def postprocess(self, result, image_shape, ori_image_shape):
        box = result['boxes']
        box[:, [0, 2]] = box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
        result['boxes'] = box
        
        if 'masks' in result:
            mask = result['masks']
            mask = paste_masks_in_image(mask, box, 1, ori_image_shape)
            result['masks'] = mask
            
        return result


def expand_detection(mask, box, padding):
    M = mask.shape[-1]
    scale = (M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 4)
    
    w_half = (box[:, 2] - box[:, 0]) * 0.5
    h_half = (box[:, 3] - box[:, 1]) * 0.5
    x_c = (box[:, 2] + box[:, 0]) * 0.5
    y_c = (box[:, 3] + box[:, 1]) * 0.5

    w_half = w_half * scale
    h_half = h_half * scale

    box_exp = Tensor.zeros_like(box)
    box_exp[:, 0] = x_c - w_half
    box_exp[:, 2] = x_c + w_half
    box_exp[:, 1] = y_c - h_half
    box_exp[:, 3] = y_c + h_half
    return padded_mask, box_exp.to(Tensor.int64)


def paste_masks_in_image(mask, box, padding, image_shape):
    mask, box = expand_detection(mask, box, padding)
    
    N = mask.shape[0]
    size = (N,) + tuple(image_shape)
    im_mask = Tensor.zeros(size, dtype=mask.dtype, device=mask.device)
    for m, b, im in zip(mask, box, im_mask):
        b = b.tolist()
        w = max(b[2] - b[0], 1)
        h = max(b[3] - b[1], 1)
        
        m = F.interpolate(m[None, None], size=(h, w), mode='bilinear', align_corners=False)[0][0]

        x1 = max(b[0], 0)
        y1 = max(b[1], 0)
        x2 = min(b[2], image_shape[1])
        y2 = min(b[3], image_shape[0])

        im[y1:y2, x1:x2] = m[(y1 - b[1]):(y2 - b[1]), (x1 - b[0]):(x2 - b[0])]
    return im_mask