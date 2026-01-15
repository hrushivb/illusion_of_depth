import argparse
import csv
import math
import os
import re
from glob import glob
from time import perf_counter
from datetime import timedelta
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unimatch.unimatch import UniMatch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_NUM_RE = re.compile(r'(\d+)')

# Fraction of the patch size to use to calculate the center region statistics in order to avoid edge artifacts
INNER_CENTER_RATIO = 0.8



def _extract_numeric_stem(fname: str):
    base = os.path.splitext(os.path.basename(fname))[0]
    m = _NUM_RE.findall(base)
    return int(m[-1]) if m else None


def _filter_by_index_range(names, start_idx: int, end_idx: int):
    sel = []
    for f in names:
        idx = _extract_numeric_stem(f)
        if idx is not None and start_idx <= idx <= end_idx:
            sel.append(f)
    return sel


def _modulo_shard(names, shard_index: int, shard_count: int, max_stem_inclusive: int = None):
    out = []
    for f in names:
        i = _extract_numeric_stem(f)
        if i is None:
            continue
        if max_stem_inclusive is not None and i > max_stem_inclusive:
            break
        if i % shard_count == shard_index:
            out.append(f)
    return out


def rgb_to_luma_uint8(img_rgb_uint8: np.ndarray) -> np.ndarray:
    r = img_rgb_uint8[..., 0].astype(np.float32)
    g = img_rgb_uint8[..., 1].astype(np.float32)
    b = img_rgb_uint8[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)


def generate_stripes(height: int, width: int, frequency: int) -> np.ndarray:
    pattern = np.zeros((height, width), dtype=np.uint8)
    stripe_w = max(1, int(frequency))
    for i in range(0, width, 2 * stripe_w):
        pattern[:, i:i + stripe_w] = 255
    return np.stack([pattern] * 3, axis=-1)


def generate_stripes_luminance(height: int, width: int, frequency: int, full_bg: np.ndarray,
                               mode: str = 'mean', contrast: float = 0.6) -> np.ndarray:
    if full_bg.shape[:2] != (height, width):
        full_bg = cv2.resize(full_bg, (width, height), interpolation=cv2.INTER_NEAREST)
    stripe_mask = np.zeros((height, width), dtype=np.uint8)
    stripe_w = max(1, int(frequency))
    for i in range(0, width, 2 * stripe_w):
        stripe_mask[:, i:i + stripe_w] = 255
    bg_luma = rgb_to_luma_uint8(full_bg).astype(np.float32)
    if mode == "local":
        bg_luma_mean_full = cv2.GaussianBlur(bg_luma, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REPLICATE)
        bg_luma_mean = cv2.resize(bg_luma_mean_full, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        bg_luma_mean = np.full((height, width), float(bg_luma.mean()), dtype=np.float32)
    amplitude = 127.0 * float(np.clip(contrast, 0.0, 1.0))
    bright = np.clip(bg_luma_mean + amplitude, 0, 255)
    dark = np.clip(bg_luma_mean - amplitude, 0, 255)
    pattern_luma = np.where(stripe_mask == 255, bright, dark).astype(np.uint8)
    pattern = np.repeat(pattern_luma[..., None], 3, axis=-1)
    return pattern


def prepare_pattern_from_image_cached(pil_img: Image.Image, ph: int, pw: int, full_bg: np.ndarray,
                                      luminance_mode: str, luminance_contrast: float,
                                      keep_aspect: bool = True) -> np.ndarray:
    img = pil_img
    orig_w, orig_h = img.size
    if keep_aspect:
        aspect = orig_w / orig_h
        pw = int(round(ph * aspect))
    img = img.resize((pw, ph), resample=Image.NEAREST)
    pat = np.array(img, dtype=np.uint8)
    if luminance_mode == 'none':
        return pat
    pat_luma = rgb_to_luma_uint8(pat).astype(np.float32)
    bg_luma = rgb_to_luma_uint8(full_bg).astype(np.float32)
    if luminance_mode == 'local':
        bg_luma_mean_full = cv2.GaussianBlur(bg_luma, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REPLICATE)
        bg_luma_mean = cv2.resize(bg_luma_mean_full, (pw, ph), interpolation=cv2.INTER_NEAREST)
    else:
        avg_bg_luma = float(bg_luma.mean())
        bg_luma_mean = np.full((ph, pw), avg_bg_luma, dtype=np.float32)
    pmin, pmax = np.min(pat_luma), np.max(pat_luma)
    if pmax - pmin < 1e-6:
        norm = np.zeros_like(pat_luma)
    else:
        norm = (pat_luma - pmin) / (pmax - pmin)
    amplitude = 127.0 * float(np.clip(luminance_contrast, 0.0, 1.0))
    out_luma = np.clip(bg_luma_mean + (norm - 0.5) * 2.0 * amplitude, 0, 255).astype(np.uint8)
    out = np.repeat(out_luma[..., None], 3, axis=-1)
    return out


def place_pattern_on_image(image: np.ndarray, pattern: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
    result = image.copy()
    ph, pw = pattern.shape[:2]
    img_h, img_w = result.shape[:2]
    y1 = max(center_y - ph // 2, 0)
    y2 = min(center_y + ph // 2, img_h)
    x1 = max(center_x - pw // 2, 0)
    x2 = min(center_x + pw // 2, img_w)
    py1 = max(0, ph // 2 - center_y)
    py2 = py1 + (y2 - y1)
    px1 = max(0, pw // 2 - center_x)
    px2 = px1 + (x2 - x1)
    if y1 >= y2 or x1 >= x2 or py1 >= py2 or px1 >= px2:
        return result
    result[y1:y2, x1:x2] = pattern[py1:py2, px1:px2]
    return result


def _extract_center_region(disp_map, center_y, center_x, region_h, region_w):

    h, w = disp_map.shape

    y1 = max(center_y - region_h // 2, 0)
    x1 = max(center_x - region_w // 2, 0)

    y2 = min(y1 + region_h, h)
    x2 = min(x1 + region_w, w)

    if y1 >= y2 or x1 >= x2:
        return disp_map

    return disp_map[y1:y2, x1:x2]


def get_center_region_stats(disp_map, center_y, center_x, region_h, region_w):
    center_region = _extract_center_region(disp_map, center_y, center_x, region_h, region_w)
    avg_disp = float(np.nanmean(center_region))
    max_disp = float(np.nanmax(center_region))
    min_disp = float(np.nanmin(center_region))
    var_disp = float(np.nanvar(center_region))
    std_disp = float(np.nanstd(center_region))
    median_disp = float(np.nanmedian(center_region))
    return avg_disp, max_disp, min_disp, var_disp, std_disp, median_disp

#Legacy function for backward compatibility
def get_center_region_stats_array(disp_map: np.ndarray, ratio: float = 0.25):
    if disp_map.ndim == 3:
        disp_map = disp_map[..., 0]
    h, w = disp_map.shape
    region_h = max(1, math.ceil(h * ratio))
    region_w = max(1, math.ceil(w * ratio))
    center_y = h // 2
    center_x = w // 2
    return get_center_region_stats(disp_map, center_y, center_x, region_h, region_w)


def center_region_histogram(disp_map, center_y, center_x, region_h, region_w, max_disp=400):
    center = _extract_center_region(disp_map, center_y, center_x, region_h, region_w)

    disp_int = np.rint(center).astype(np.int32)
    disp_int = np.clip(disp_int, 0, max_disp)
    hist = np.bincount(disp_int.ravel(), minlength=max_disp + 1)
    return hist


def compute_d1_score(disp_map, center_y, center_x, region_h, region_w, ground_truth_disp, threshold=3.0):
    center_region = _extract_center_region(disp_map, center_y, center_x, region_h, region_w)
    
    # Compute absolute error: |predicted - ground_truth|
    error = np.abs(center_region - ground_truth_disp)
    
    # Count pixels with error > threshold
    total_pixels = center_region.size
    bad_pixels = int((error > threshold).sum())
    
    d1_score = bad_pixels / float(max(1, total_pixels))
    return d1_score


def load_model(args, device):
    model = UniMatch(
        feature_channels=args.feature_channels,
        num_scales=args.num_scales,
        upsample_factor=args.upsample_factor,
        num_head=args.num_head,
        ffn_dim_expansion=args.ffn_dim_expansion,
        num_transformer_layers=args.num_transformer_layers,
        reg_refine=args.reg_refine,
        task=args.task
    ).to(device)
    checkpoint = torch.load(args.load_model, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


def list_common_pairs(
    left_dir: str,
    right_dir: str,
    max_pairs: int,
    shard_index: int,
    shard_count: int,
    index_range: str = '',
    shard_mode: str = 'mod',
    max_stem: int = None,
):
    lefts = sorted([f for f in os.listdir(left_dir)])
    rights = set([f for f in os.listdir(right_dir)])
    commons = [f for f in lefts if f in rights]
    if index_range:
        try:
            start_s, end_s = index_range.split('-', 1)
            start_idx = int(start_s.strip()); end_idx = int(end_s.strip())
            commons = _filter_by_index_range(commons, start_idx, end_idx)
            print(f"[info] index_range {start_idx}-{end_idx}: now {len(commons)} files")
        except Exception as e:
            print(f"[warn] invalid --index_range '{index_range}': {e} (ignored)")
    if shard_count and shard_count > 1 and shard_index >= 0 and shard_mode == 'mod':
        before = len(commons)
        commons = _modulo_shard(commons, shard_index, shard_count, max_stem_inclusive=max_stem)
        print(f"[info] modulo shard {shard_index}/{shard_count}: {before} -> {len(commons)} files"
              + (f" (max_stem={max_stem})" if max_stem is not None else ""))
    else:
        if max_stem is not None:
            capped = []
            for f in commons:
                i = _extract_numeric_stem(f)
                if i is None:
                    continue
                if i > max_stem:
                    break
                capped.append(f)
            commons = capped
    return commons[:max_pairs]


def load_external_patterns(pattern_dir: str):
    patterns = []
    if not pattern_dir or not os.path.isdir(pattern_dir):
        return patterns
    size_dirs = [d for d in glob(os.path.join(pattern_dir, 'size_*')) if os.path.isdir(d)]
    for sd in sorted(size_dirs):
        m = re.match(r'^size_(\d+)$', os.path.basename(sd))
        freq = int(m.group(1)) if m else None
        if freq is None:
            continue
        img_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
            img_paths.extend(glob(os.path.join(sd, ext)))
        for ip in sorted(img_paths):
            gran = os.path.splitext(os.path.basename(ip))[0]
            fname = os.path.basename(ip)
            patterns.append({'fname': fname, 'freq': int(freq), 'granularity': gran, 'path': ip})
    return patterns


def _gnorm(g):
    if g is None:
        return None
    try:
        return str(int(str(g)))
    except Exception:
        return str(g)


def _index_by_fg(patterns):
    idx = {}
    for p in patterns:
        f = int(p['freq'])
        g = _gnorm(p['granularity'])
        idx.setdefault((f, g), []).append(p)
    return idx


def parse_fg_list(s):
    pairs = []
    if not s:
        return pairs
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' in tok:
            f_str, g_str = tok.split(':', 1)
            pairs.append((int(f_str), _gnorm(g_str.strip())))
        else:
            pairs.append((int(tok), None))
    return pairs

K_L_BASE = np.array([
    [1052.72, 0, 1100],
    [0, 1052.13, 624.116],
    [0, 0, 1]
], dtype=np.float64)
D_L_BASE = np.array([-0.0414982, 0.00986681, 0.000350913, -0.000128067, -0.00492415], dtype=np.float64)

K_R_BASE = np.array([
    [1057.44, 0, 1099.21],
    [0, 1056.78, 603.911],
    [0, 0, 1]
], dtype=np.float64)
D_R_BASE = np.array([-0.0423852, 0.0111663, 0.000146176, 9.69267e-05, -0.005089], dtype=np.float64)


def apply_calibration_error(K, D, magnitude: int):
    scale = magnitude / 4.0
    K_modified = K.copy()
    D_modified = D.copy() * scale

    focal_error = (magnitude - 4) * 0.02
    principal_error = (magnitude - 4) * 5.0

    K_modified[0, 0] *= (1.0 + focal_error)
    K_modified[1, 1] *= (1.0 + focal_error)
    K_modified[0, 2] += principal_error
    K_modified[1, 2] += principal_error

    return K_modified, D_modified


def apply_distortion_to_ideal_image(img_ideal: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    h, w = img_ideal.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)
    img_distorted = cv2.remap(img_ideal, map_x, map_y, cv2.INTER_LINEAR)
    return img_distorted


#Unimatch inference function
def _to_normalized_tensor(np_uint8_hw3: np.ndarray, device, pin: bool, non_blocking: bool):
    x = torch.from_numpy(np_uint8_hw3).permute(2, 0, 1).contiguous().float().div(255.0)
    mean = torch.as_tensor(IMAGENET_MEAN[:, None, None])
    std = torch.as_tensor(IMAGENET_STD[:, None, None])
    x = x.sub(mean).div(std)
    if device.type == 'cuda' and pin:
        x = x.pin_memory()
        x = x.to(device, non_blocking=non_blocking)
    else:
        x = x.to(device)
    return x


@torch.no_grad()
def run_unimatch_np(model,
                    left_np_uint8: np.ndarray,
                    right_np_uint8: np.ndarray,
                    device,
                    padding_factor=32,
                    inference_size=None,
                    attn_type=None,
                    attn_splits_list=None,
                    corr_radius_list=None,
                    prop_radius_list=None,
                    num_reg_refine=3,
                    use_amp=False,
                    non_blocking=True,
                    pin_memory=True):
    left_t = _to_normalized_tensor(left_np_uint8, device, pin_memory, non_blocking).unsqueeze(0)
    right_t = _to_normalized_tensor(right_np_uint8, device, pin_memory, non_blocking).unsqueeze(0)

    nearest_size = [
        int(np.ceil(left_t.size(-2) / padding_factor)) * padding_factor,
        int(np.ceil(left_t.size(-1) / padding_factor)) * padding_factor
    ]
    use_size = nearest_size if inference_size is None else list(inference_size)
    ori_h, ori_w = left_t.shape[-2:]

    if use_size[0] != ori_h or use_size[1] != ori_w:
        left_t = F.interpolate(left_t, size=use_size, mode='bilinear', align_corners=True)
        right_t = F.interpolate(right_t, size=use_size, mode='bilinear', align_corners=True)

    if use_amp:
        if device.type == 'cuda':
            autocast_ctx = torch.cuda.amp.autocast
        else:
            autocast_ctx = lambda **kwargs: torch.amp.autocast('cpu', **kwargs)
    else:
        from contextlib import nullcontext
        autocast_ctx = lambda **kwargs: nullcontext()
    with autocast_ctx(enabled=use_amp):
        pred_disp = model(left_t, right_t,
                          attn_type=attn_type,
                          attn_splits_list=attn_splits_list,
                          corr_radius_list=corr_radius_list,
                          prop_radius_list=prop_radius_list,
                          num_reg_refine=num_reg_refine,
                          task='stereo')['flow_preds'][-1]

    if use_size[0] != ori_h or use_size[1] != ori_w:
        pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=(ori_h, ori_w),
                                  mode='bilinear', align_corners=True).squeeze(1)
        pred_disp = pred_disp * (ori_w / float(use_size[-1]))

    disp_np = pred_disp[0].detach().float().cpu().numpy().astype(np.float32)
    return disp_np


class ProgTimer:
    def __init__(self):
        self.count = 0
        self.total = 0
        self.sum_dt = 0.0
    def update(self, dt):
        self.count += 1
        self.sum_dt += dt
    @property
    def avg_dt(self):
        return (self.sum_dt / self.count) if self.count > 0 else None
    def eta(self):
        if self.count == 0 or self.avg_dt is None or self.total <= self.count:
            return None
        remain = self.total - self.count
        secs = remain * self.avg_dt
        return remain, timedelta(seconds=int(secs))

# NOTE: the notation used in the code does not match the notation used in the paper.
# The code uses the notation "Granularity" for the size of the black sqare within the checkerboard pattern, while the paper uses the notation "patch size".
# The code uses the notation "Frequency" for the frequency of the checkerboard pattern, while the paper uses the notation "Granularity".
def main():
    parser = argparse.ArgumentParser("Attack Patch Evaluation (pattern-first, in-memory, accurate ETA)")

    parser.add_argument('--left_dir', type=str, required=True)
    parser.add_argument('--right_dir', type=str, required=True)

    parser.add_argument('--ratios', type=str, default='0.25')
    parser.add_argument('--frequency', type=str, default='2,4,8')
    parser.add_argument('--discrepency', type=str, default='10')

    parser.add_argument('--max_pairs', type=int, default=10)
    parser.add_argument('--index_range', type=str, default='')
    parser.add_argument('--shard_index', type=int, default=-1)
    parser.add_argument('--shard_count', type=int, default=-1)
    parser.add_argument('--shard_mode', type=str, default='mod', choices=['block', 'mod'])
    parser.add_argument('--max_stem', type=int, default=None)

    parser.add_argument('--load_model', type=str, required=True)
    parser.add_argument('--feature_channels', type=int, default=128)
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--upsample_factor', type=int, default=4)
    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--ffn_dim_expansion', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--reg_refine', action='store_true')
    parser.add_argument('--task', type=str, default='stereo')

    parser.add_argument('--attn_type', type=str, default='self_swin2d_cross_swin1d')
    parser.add_argument('--attn_splits_list', type=int, nargs='+', default=[2, 8])
    parser.add_argument('--corr_radius_list', type=int, nargs='+', default=[-1, 4])
    parser.add_argument('--prop_radius_list', type=int, nargs='+', default=[-1, 1])
    parser.add_argument('--num_reg_refine', type=int, default=3)

    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--save_pfm', action='store_true', default=False)
    parser.add_argument('--save_raw_disparity', action='store_true', default=False)
    parser.add_argument('--save_disparity_maps', action='store_true', default=False)

    parser.add_argument('--padding_factor', type=int, default=32)
    parser.add_argument('--inference_size', nargs='+', type=int, default=[512, 704])

    parser.add_argument('--luminance', type=str, default='none', choices=['none', 'mean', 'local'])
    parser.add_argument('--luminance_contrast', type=float, default=0.6)
    parser.add_argument('--max_save_pairs', type=int, default=10)

    parser.add_argument('--pattern_dir', type=str, default='')
    parser.add_argument('--pattern_dir_left', type=str, default='')
    parser.add_argument('--pattern_dir_right', type=str, default='')

    parser.add_argument('--csv_name', type=str, default='results.csv')
    parser.add_argument('--fg_list', type=str, default='', help='Comma-separated freq:gran pairs, e.g., "50:50,09:09"; normalized')

    parser.add_argument('--eta_include_benign', action='store_true', help='Include benign inference calls in ETA calculation')
    parser.add_argument('--amp', action='store_true', help='Enable autocast (mixed precision) for inference')
    parser.add_argument('--cudnn_benchmark', action='store_true', help='Enable cudnn benchmark for speed (non-deterministic)')

    parser.add_argument('--flush_every', type=int, default=100, help='Flush CSV every N rows')
    
    parser.add_argument('--calib_error_magnitudes', type=str, default='0',
                        help='Comma-separated list of calibration error magnitudes (0,1-5). 0 = none.')
    parser.add_argument('--distort_benign', action='store_true', default=False,
                        help='If set, apply calibration error to benign images (per calib magnitude) and save distorted-benign stats/images.')
    parser.add_argument('--full_pattern', action='store_true', default=False,
                        help='Use pattern at native size (no scaling); crop height to 75% if needed')
    parser.add_argument('--keep_aspect', action='store_true', default=False,
                        help='Keep aspect ratio when resizing patterns')

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = not args.cudnn_benchmark
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    ratio_list = [float(r) for r in args.ratios.split(',') if r.strip()]
    freq_list = [int(f) for f in args.frequency.split(',') if f.strip()]
    disp_list = [int(d) for d in args.discrepency.split(',') if d.strip()]
    calib_mag_list = [int(x) for x in args.calib_error_magnitudes.split(',') if x.strip()]

    os.makedirs(args.results_dir, exist_ok=True)
    artifacts_root = os.path.join(args.results_dir, "artifacts")
    os.makedirs(artifacts_root, exist_ok=True)

    
    # Create separate subdirectories for different artifact types
    disparity_maps_root = os.path.join(artifacts_root, "disparity_maps")
    input_images_root = os.path.join(artifacts_root, "input_images")
    os.makedirs(disparity_maps_root, exist_ok=True)
    os.makedirs(input_images_root, exist_ok=True)
    
    # Subdirectories for input images
    benign_images_dir = os.path.join(input_images_root, "benign")
    distorted_benign_images_dir = os.path.join(input_images_root, "distorted_benign")
    attack_images_dir = os.path.join(input_images_root, "attack")
    os.makedirs(benign_images_dir, exist_ok=True)
    os.makedirs(distorted_benign_images_dir, exist_ok=True)
    os.makedirs(attack_images_dir, exist_ok=True)
    
    # Legacy: keep benign_pairs_root for backward compatibility. Not used presently.
    benign_pairs_root = os.path.join(artifacts_root, "benign_pairs")
    os.makedirs(benign_pairs_root, exist_ok=True)

    csv_path = os.path.join(args.results_dir, args.csv_name)
    fieldnames = [
        'pair', 'ratio', 'frequency', 'granularity', 'discrepency', 'calib_magnitude',
        'mean_benign', 'mean_attack', 'mean_diff',
        'max_benign', 'max_attack', 'max_diff',
        'min_benign', 'min_attack', 'min_diff',
        'var_benign', 'var_attack', 'var_diff',
        'std_benign', 'std_attack', 'std_diff',
        'median_benign', 'median_attack', 'median_diff',
        # Inner-center (inner patch) stats
        'mean_benign_inner', 'mean_attack_inner', 'mean_diff_inner',
        'max_benign_inner', 'max_attack_inner', 'max_diff_inner',
        'min_benign_inner', 'min_attack_inner', 'min_diff_inner',
        'var_benign_inner', 'var_attack_inner', 'var_diff_inner',
        'std_benign_inner', 'std_attack_inner', 'std_diff_inner',
        'median_benign_inner', 'median_attack_inner', 'median_diff_inner',
        # D1 scores (using fixed discrepancy as ground truth)
        'd1_full', 'd1_inner',
        'mean_benign_dist', 'max_benign_dist', 'min_benign_dist',
        'var_benign_dist', 'std_benign_dist', 'median_benign_dist'
    ]
    
    # Histogram CSVs
    hist_full_csv_path = os.path.join(args.results_dir, 'hist_center_region_full.csv')
    hist_inner_csv_path = os.path.join(args.results_dir, 'hist_center_region_inner.csv')
    
    hist_fields = ['pair', 'ratio', 'frequency', 'granularity', 'discrepency', 'calib_magnitude', 'phase']
    hist_fields.extend([f'd{d}' for d in range(401)])

    from collections import defaultdict
    import shutil
    from datetime import datetime

    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    # Histogram CSV setup
    hist_full_exists = os.path.exists(hist_full_csv_path) and os.path.getsize(hist_full_csv_path) > 0
    hist_inner_exists = os.path.exists(hist_inner_csv_path) and os.path.getsize(hist_inner_csv_path) > 0

    csv_f = None
    w = None
    hist_full_f = None
    hist_inner_f = None
    hist_writer_full = None
    hist_writer_inner = None

    done_rows = set()  # (pair, ratio, freq, gran, disp, calib_mag)
    done_pairs_per_combo = defaultdict(set)  # (ratio, freq, gran, disp, calib_mag) -> {pair}

    # Histogram resume indices (de-dupe)
    # Key: (pair, ratio, frequency, granularity, discrepency, calib_magnitude, phase)
    done_hist_full = set()
    done_hist_inner = set()

    def _norm_hist_gran(x):
        if x is None:
            return ''
        s = str(x)
        if s in ('', 'None'):
            return ''
        return s

    def _hist_key_from_row(row):
        try:
            pair = row.get('pair', '')
            ratio = float(row.get('ratio'))
            freq = int(row.get('frequency'))
            gran = _norm_hist_gran(row.get('granularity'))
            disp = int(row.get('discrepency'))
            calib_mag = int(row.get('calib_magnitude', 0))
            phase = row.get('phase', '')
        except Exception:
            return None
        return (pair, ratio, freq, gran, disp, calib_mag, phase)

    if csv_exists:
        # Keep a single rolling backup to avoid accumulating many timestamped backups.
        backup_path = csv_path + ".bak"
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
            shutil.copy2(csv_path, backup_path)
            print(f"[safety] Backed up existing CSV to {backup_path}")
        except Exception as e:
            print(f"[warn] Failed to back up CSV ({e}); proceeding without backup")

        with open(csv_path, 'r', newline='') as f_in:
            reader = csv.DictReader(f_in)
            row_count = 0
            for row in reader:
                row_count += 1
                try:
                    pair = row['pair']
                    ratio = float(row['ratio'])
                    freq = int(row['frequency'])
                    gran = row['granularity'] if row['granularity'] not in ('', 'None') else None
                    disp = int(row['discrepency'])
                    calib_mag = int(row.get('calib_magnitude', 0))
                except Exception as e:
                    continue
                key_row = (pair, ratio, freq, gran, disp, calib_mag)
                key_combo = (ratio, freq, gran, disp, calib_mag)
                done_rows.add(key_row)
                done_pairs_per_combo[key_combo].add(pair)

        print(f"[resume] Loaded {len(done_rows)} completed rows from existing CSV (read {row_count} total rows)")
    else:
        # print(f"[DEBUG] CSV does not exist, starting fresh")
        pass

    # Load existing histogram rows so we don't append duplicates on resume/rerun.
    if hist_full_exists:
        try:
            with open(hist_full_csv_path, 'r', newline='') as hf:
                hr = csv.DictReader(hf)
                for row in hr:
                    k = _hist_key_from_row(row)
                    if k is not None:
                        done_hist_full.add(k)
            print(f"[resume] Loaded {len(done_hist_full)} completed rows from existing full histogram CSV")
        except Exception as e:
            print(f"[warn] Failed to load full histogram CSV for resume ({e}); will append without de-dupe")

    if hist_inner_exists:
        try:
            with open(hist_inner_csv_path, 'r', newline='') as hf:
                hr = csv.DictReader(hf)
                for row in hr:
                    k = _hist_key_from_row(row)
                    if k is not None:
                        done_hist_inner.add(k)
            print(f"[resume] Loaded {len(done_hist_inner)} completed rows from existing inner histogram CSV")
        except Exception as e:
            print(f"[warn] Failed to load inner histogram CSV for resume ({e}); will append without de-dupe")

    try:
        hist_full_f = open(hist_full_csv_path, 'a', newline='')
        hist_writer_full = csv.DictWriter(hist_full_f, fieldnames=hist_fields)
        if not hist_full_exists:
            hist_writer_full.writeheader()

        hist_inner_f = open(hist_inner_csv_path, 'a', newline='')
        hist_writer_inner = csv.DictWriter(hist_inner_f, fieldnames=hist_fields)
        if not hist_inner_exists:
            hist_writer_inner.writeheader()

        csv_f = open(csv_path, 'a', newline='')
        w = csv.DictWriter(csv_f, fieldnames=fieldnames)
        if not csv_exists:
            w.writeheader()

        pairs = list_common_pairs(
            args.left_dir, args.right_dir, args.max_pairs,
            shard_index=args.shard_index, shard_count=args.shard_count,
            index_range=args.index_range,
            shard_mode=args.shard_mode, max_stem=args.max_stem
        )
        print(f"[info] final selected pairs: {len(pairs)}")

        fg_explicit = parse_fg_list(args.fg_list)

        def build_pattern_iter():
            items = []

            # Explicit stereo patterns with L/R dirs
            if fg_explicit:
                if args.pattern_dir_left and args.pattern_dir_right:
                    L_all = load_external_patterns(args.pattern_dir_left)
                    R_all = load_external_patterns(args.pattern_dir_right)
                    L_idx = _index_by_fg(L_all)
                    R_idx = _index_by_fg(R_all)

                    missing = []
                    for f, g in fg_explicit:
                        key = (f, g)
                        if key not in L_idx or key not in R_idx:
                            reason = []
                            if key not in L_idx:
                                reason.append('L-missing')
                            if key not in R_idx:
                                reason.append('R-missing')
                            l_any = any(k[0] == f for k in L_idx.keys())
                            r_any = any(k[0] == f for k in R_idx.keys())
                            if l_any and r_any:
                                reason.append('freq-only available')
                            missing.append((f, g, ','.join(reason)))
                    if missing:
                        print("[warn] Explicit pairs not matched (after zero-pad normalization):")
                        for f, g, why in missing:
                            print(f"  freq={f}, gran={g} -> {why}")

                    for f, g in fg_explicit:
                        key = (f, g)
                        if key in L_idx and key in R_idx:
                            items.append({
                                'freq': f, 'granularity': g,
                                'pathL': L_idx[key][0]['path'],
                                'pathR': R_idx[key][0]['path'],
                                'stereo': True, 'external': True
                            })
                        else:
                            L_any = next((p for k, vs in L_idx.items() if k[0] == f for p in vs), None)
                            R_any = next((p for k, vs in R_idx.items() if k[0] == f for p in vs), None)
                            if L_any and R_any:
                                items.append({
                                    'freq': f,
                                    'granularity': _gnorm(L_any['granularity']),
                                    'pathL': L_any['path'],
                                    'pathR': R_any['path'],
                                    'stereo': True, 'external': True
                                })
                    if items:
                        return items

                if args.pattern_dir:
                    P_all = load_external_patterns(args.pattern_dir)
                    P_idx = _index_by_fg(P_all)
                    for f, g in fg_explicit:
                        key = (f, g)
                        if key in P_idx:
                            items.append({
                                'freq': f, 'granularity': g,
                                'path': P_idx[key][0]['path'],
                                'stereo': False, 'external': True
                            })
                        else:
                            P_any = next((p for k, vs in P_idx.items() if k[0] == f for p in vs), None)
                            if P_any:
                                items.append({
                                    'freq': f,
                                    'granularity': _gnorm(P_any['granularity']),
                                    'path': P_any['path'],
                                    'stereo': False, 'external': True
                                })
                    if items:
                        return items

                for f, g in fg_explicit:
                    items.append({
                        'freq': f, 'granularity': g,
                        'path': None, 'stereo': False, 'external': False
                    })
                return items

            if args.pattern_dir_left and args.pattern_dir_right:
                L_all = load_external_patterns(args.pattern_dir_left)
                R_all = load_external_patterns(args.pattern_dir_right)
                L_idx = _index_by_fg(L_all)
                R_idx = _index_by_fg(R_all)
                stereo_specs = sorted(set(L_idx.keys()) & set(R_idx.keys()))
                for (f, g) in stereo_specs:
                    items.append({
                        'freq': f, 'granularity': g,
                        'pathL': L_idx[(f, g)][0]['path'],
                        'pathR': R_idx[(f, g)][0]['path'],
                        'stereo': True, 'external': True
                    })
                if items:
                    return items

            if args.pattern_dir:
                P_all = load_external_patterns(args.pattern_dir)
                P_idx = _index_by_fg(P_all)
                for (f, g), vs in sorted(P_idx.items()):
                    items.append({
                        'freq': f, 'granularity': g,
                        'path': vs[0]['path'],
                        'stereo': False, 'external': True
                    })
                if items:
                    return items

            for f in freq_list:
                items.append({
                    'freq': f, 'granularity': None,
                    'path': None, 'stereo': False, 'external': False
                })
            return items

        pattern_items = build_pattern_iter()

        def _src_desc(item):
            if item.get('stereo', False) and item.get('external', False):
                return f"L:{os.path.basename(item['pathL'])}, R:{os.path.basename(item['pathR'])}"
            if (not item.get('stereo', False)) and item.get('external', False):
                return os.path.basename(item['path'])
            return "generated"

        print(f"[info] Patterns to test: {len(pattern_items)}")
        for k, it in enumerate(pattern_items):
            f = it.get('freq'); g = it.get('granularity')
            stereo = it.get('stereo', False); external = it.get('external', False)
            src = _src_desc(it)
            print(f" [{k:02d}] freq={f}, gran={g}, stereo={stereo}, external={external}, src={src}")

        pt = ProgTimer()

        total_combos = len(ratio_list) * len(pattern_items) * len(disp_list) * len(calib_mag_list) * len(pairs)
        already_done = len(done_rows)
        remaining_rows = max(total_combos - already_done, 0)

        attack_total = remaining_rows
        benign_total = len(ratio_list) * len(pairs) if args.eta_include_benign else 0
        if args.distort_benign:
            benign_total += len(ratio_list) * len(calib_mag_list) * len(pairs) if args.eta_include_benign else 0
        pt.total = attack_total + benign_total

        print(f"[progress] planned remaining inferences = {pt.total} "
              f"(attack_remaining={attack_total}"
              f"{', benign='+str(benign_total) if benign_total else ''})")

        rows_since_flush = 0
        skipped_pattern_combos = 0

        @lru_cache(maxsize=None)
        def load_pil(path):
            return Image.open(path).convert('RGB')

        for ratio in ratio_list:
            print(f"[info] Processing ratio {ratio:.3f}")

            # BENIGN PASS
            benign_stats_cache = {}
            benign_inner_stats_cache = {}
            # Per-ratio, per-image region metadata based on patch size on the original image
            # center_region_meta[fname] = (cy, cx, ph, pw, inner_ph, inner_pw)
            center_region_meta = {}
            
            for idx, fname in enumerate(pairs):
                # Start timing from the beginning of benign unit processing
                t_benign_start = perf_counter()
                
                left_path = os.path.join(args.left_dir, fname)
                right_path = os.path.join(args.right_dir, fname)
                try:
                    limg = np.array(Image.open(left_path).convert("RGB"))
                    rimg = np.array(Image.open(right_path).convert("RGB"))
                except Exception as e:
                    print(f"Skipping {fname}: failed to load images ({e})")
                    continue

                h_img, w_img = limg.shape[:2]
                ph = int(math.ceil(h_img * ratio))
                pw = int(math.ceil(w_img * ratio))
                cy = h_img // 2
                cx = w_img // 2
                inner_ph = max(1, int(round(ph * INNER_CENTER_RATIO)))
                inner_pw = max(1, int(round(pw * INNER_CENTER_RATIO)))
                center_region_meta[fname] = (cy, cx, ph, pw, inner_ph, inner_pw)

                # Save benign images right before inference
                if args.save_raw_disparity or args.save_disparity_maps:
                    stem, _ext = os.path.splitext(fname)
                    try:
                        # Save to input_images/benign subdirectory
                        benign_subdir = os.path.join(benign_images_dir, f"{stem}_ratio{ratio:.3f}")
                        os.makedirs(benign_subdir, exist_ok=True)
                        
                        # Save left and right benign images
                        benign_left_path = os.path.join(benign_subdir, f"{stem}_left.png")
                        benign_right_path = os.path.join(benign_subdir, f"{stem}_right.png")
                        # print(f"[DEBUG] Saving benign images: left={benign_left_path}, right={benign_right_path}")
                        Image.fromarray(limg).save(benign_left_path)
                        Image.fromarray(rimg).save(benign_right_path)
                        # print(f"[DEBUG] Benign images saved. Left exists: {os.path.exists(benign_left_path)}, Right exists: {os.path.exists(benign_right_path)}")
                    except Exception as e:
                        import traceback
                        print(f"[ERROR] Failed to save benign images for {fname}: {e}")
                        traceback.print_exc()

                benign_disp = run_unimatch_np(
                    model, limg, rimg, device,
                    padding_factor=args.padding_factor,
                    inference_size=args.inference_size,
                    attn_type=args.attn_type,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list,
                    num_reg_refine=args.num_reg_refine,
                    use_amp=args.amp
                )

                cy, cx, ph, pw, inner_ph, inner_pw = center_region_meta[fname]
                
                # Full-patch center region stats
                stats_center = get_center_region_stats(benign_disp, cy, cx, ph, pw)
                benign_stats_cache[fname] = stats_center

                # Inner-patch center region stats
                stats_inner = get_center_region_stats(benign_disp, cy, cx, inner_ph, inner_pw)
                benign_inner_stats_cache[fname] = stats_inner

                # benign histogram over full patch region
                hist = center_region_histogram(benign_disp, cy, cx, ph, pw, max_disp=400)
                hist_row = {
                    'pair': fname,
                    'ratio': ratio,
                    'frequency': -1,
                    'granularity': '',
                    'discrepency': -1,
                    'calib_magnitude': 0,
                    'phase': 'benign',
                }
                for d in range(401):
                    hist_row[f'd{d}'] = int(hist[d])
                k_full = (fname, float(ratio), int(-1), _norm_hist_gran(''), int(-1), int(0), 'benign')
                if k_full not in done_hist_full:
                    hist_writer_full.writerow(hist_row)
                    hist_full_f.flush()
                    done_hist_full.add(k_full)

                # benign inner-center histogram
                hist_inner = center_region_histogram(benign_disp, cy, cx, inner_ph, inner_pw, max_disp=400)
                hist_inner_row = {
                    'pair': fname,
                    'ratio': ratio,
                    'frequency': -1,
                    'granularity': '',
                    'discrepency': -1,
                    'calib_magnitude': 0,
                    'phase': 'benign',
                }
                for d in range(401):
                    hist_inner_row[f'd{d}'] = int(hist_inner[d])
                k_inner = (fname, float(ratio), int(-1), _norm_hist_gran(''), int(-1), int(0), 'benign')
                if k_inner not in done_hist_inner:
                    hist_writer_inner.writerow(hist_inner_row)
                    hist_inner_f.flush()
                    done_hist_inner.add(k_inner)

                # Save benign disparity maps if requested
                if args.save_raw_disparity or args.save_disparity_maps:
                    stem, _ext = os.path.splitext(fname)
                    try:
                        benign_disp_dir = os.path.join(disparity_maps_root, "benign", f"{stem}_ratio{ratio:.3f}")
                        os.makedirs(benign_disp_dir, exist_ok=True)
                        
                        if args.save_raw_disparity:
                            benign_raw_path = os.path.join(benign_disp_dir, f"{stem}_benign.npy")
                            np.save(benign_raw_path, benign_disp)
                        else:
                            pass
                        
                        if args.save_disparity_maps:
                            dm_benign = np.array(benign_disp, dtype=np.float32)
                            finite_mask_benign = np.isfinite(dm_benign)
                            finite_count_benign = np.sum(finite_mask_benign)
                            total_count_benign = dm_benign.size
                            # print(f"[DEBUG] Benign finite values: {finite_count_benign}/{total_count_benign} ({100*finite_count_benign/total_count_benign:.2f}%)")
                            
                            if not np.any(finite_mask_benign):
                                disp_norm_benign = np.zeros_like(dm_benign, dtype=np.uint8)
                            else:
                                dmin_benign = float(np.nanmin(dm_benign))
                                dmax_benign = float(np.nanmax(dm_benign))
                                denom_benign = max(dmax_benign - dmin_benign, 1e-8)
                                # print(f"[DEBUG] Normalizing benign: dmin={dmin_benign:.3f}, dmax={dmax_benign:.3f}, denom={denom_benign:.3f}")
                                disp_norm_benign = ((dm_benign - dmin_benign) / denom_benign * 255.0).astype(np.uint8)
                                # print(f"[DEBUG] Benign normalized image stats: min={disp_norm_benign.min()}, max={disp_norm_benign.max()}, mean={disp_norm_benign.mean():.2f}")
                            
                            benign_png_path = os.path.join(benign_disp_dir, f"{stem}_benign.png")
                            Image.fromarray(disp_norm_benign).save(benign_png_path)
                            # print(f"[DEBUG] Benign disparity map saved. File exists: {os.path.exists(benign_png_path)}, size={os.path.getsize(benign_png_path) if os.path.exists(benign_png_path) else 'N/A'} bytes")
                        else:
                            # print(f"[DEBUG] Skipping benign disparity map save (flag=False)")
                            pass
                        
                        # print(f"[DEBUG] Benign disparity map saving completed successfully for {fname}")
                    except Exception as e:
                        import traceback
                        print(f"[ERROR] Failed to save benign disparity maps for {fname}: {e}")
                        print(f"[ERROR] Traceback:")
                        traceback.print_exc()

                # End timing after all benign processing (I/O, inference, stats, histogram writing)
                dt_benign = perf_counter() - t_benign_start
                if args.eta_include_benign:
                    pt.update(dt_benign)
                    eta_b = pt.eta()
                    if eta_b is not None:
                        remain_b, eta_td_b = eta_b
                        print(f"[timing] benign pair={fname} dt={dt_benign*1000:.2f} ms | "
                              f"avg={pt.avg_dt*1000:.2f} ms | remaining={remain_b} | ETA={eta_td_b}")
                    else:
                        print(f"[timing] benign pair={fname} dt={dt_benign*1000:.2f} ms | "
                              f"avg=warming | remaining={pt.total - pt.count}")

            # ATTACK PASS over calib magnitudes
            for calib_mag in calib_mag_list:
                if calib_mag == 0:
                    use_calib = False
                    K_L_err = D_L_err = K_R_err = D_R_err = None
                    print(f"[calib] Running attacks with calib_magnitude=0 (no calibration error)")
                else:
                    use_calib = True
                    K_L_err, D_L_err = apply_calibration_error(K_L_BASE, D_L_BASE, calib_mag)
                    K_R_err, D_R_err = apply_calibration_error(K_R_BASE, D_R_BASE, calib_mag)
                    print(f"[calib] Running attacks with calib_magnitude={calib_mag}")

                # Optional distorted-benign pass per calib - unused presently.
                distorted_benign_stats_cache = {}
                if args.distort_benign and use_calib:
                    print(f"[calib] Computing distorted-benign disparity for calib_magnitude={calib_mag}")
                    for idx, fname in enumerate(pairs):
                        # Start timing from the beginning of distorted-benign unit processing
                        t_dist_benign_start = perf_counter()
                        
                        left_path = os.path.join(args.left_dir, fname)
                        right_path = os.path.join(args.right_dir, fname)
                        try:
                            limg = np.array(Image.open(left_path).convert("RGB"))
                            rimg = np.array(Image.open(right_path).convert("RGB"))
                        except Exception as e:
                            print(f"Skipping {fname}: failed to load images ({e})")
                            continue

                        l_dist = apply_distortion_to_ideal_image(limg, K_L_err, D_L_err)
                        r_dist = apply_distortion_to_ideal_image(rimg, K_R_err, D_R_err)

                        # Save distorted-benign images right before inference
                        if args.save_raw_disparity or args.save_disparity_maps:
                            stem, _ext = os.path.splitext(fname)
                            try:
                                # Save to input_images/distorted_benign subdirectory
                                dist_benign_subdir = os.path.join(distorted_benign_images_dir, f"{stem}_ratio{ratio:.3f}_calibM{calib_mag}")
                                os.makedirs(dist_benign_subdir, exist_ok=True)
                                
                                # Save left and right distorted-benign images
                                dist_benign_left_path = os.path.join(dist_benign_subdir, f"{stem}_left.png")
                                dist_benign_right_path = os.path.join(dist_benign_subdir, f"{stem}_right.png")
                                # print(f"[DEBUG] Saving distorted-benign images: left={dist_benign_left_path}, right={dist_benign_right_path}")
                                Image.fromarray(l_dist).save(dist_benign_left_path)
                                Image.fromarray(r_dist).save(dist_benign_right_path)
                                # print(f"[DEBUG] Distorted-benign images saved. Left exists: {os.path.exists(dist_benign_left_path)}, Right exists: {os.path.exists(dist_benign_right_path)}")
                            except Exception as e:
                                import traceback
                                print(f"[ERROR] Failed to save distorted-benign images for {fname}: {e}")
                                traceback.print_exc()

                        disp_unp = run_unimatch_np(
                            model, l_dist, r_dist, device,
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            attn_type=args.attn_type,
                            attn_splits_list=args.attn_splits_list,
                            corr_radius_list=args.corr_radius_list,
                            prop_radius_list=args.prop_radius_list,
                            num_reg_refine=args.num_reg_refine,
                            use_amp=args.amp
                        )

                        # Use the same center region as benign for this ratio/fname
                        cy, cx, ph, pw, inner_ph, inner_pw = center_region_meta[fname]
                        stats_dist = get_center_region_stats(disp_unp, cy, cx, ph, pw)
                        distorted_benign_stats_cache[fname] = stats_dist

                        # Save distorted-benign disparity maps if requested
                        if args.save_raw_disparity or args.save_disparity_maps:
                            stem, _ext = os.path.splitext(fname)
                            # print(f"[DEBUG] Saving distorted-benign disparity maps for {fname}, calib_mag={calib_mag}")
                            try:
                                # Save to disparity_maps/distorted_benign subdirectory
                                dist_benign_disp_dir = os.path.join(disparity_maps_root, "distorted_benign", f"{stem}_ratio{ratio:.3f}_calibM{calib_mag}")
                                # print(f"[DEBUG] Creating distorted-benign disparity directory: {dist_benign_disp_dir}")
                                os.makedirs(dist_benign_disp_dir, exist_ok=True)
                                # print(f"[DEBUG] Distorted-benign disparity directory created/exists: {os.path.exists(dist_benign_disp_dir)}")
                                
                                if args.save_raw_disparity:
                                    dist_benign_raw_path = os.path.join(dist_benign_disp_dir, f"{stem}_distorted_benign.npy")
                                    # print(f"[DEBUG] Saving distorted-benign raw disparity to: {dist_benign_raw_path}")
                                    # print(f"[DEBUG] disp_unp stats: min={np.nanmin(disp_unp):.3f}, max={np.nanmax(disp_unp):.3f}, mean={np.nanmean(disp_unp):.3f}")
                                    np.save(dist_benign_raw_path, disp_unp)
                                    # print(f"[DEBUG] Distorted-benign raw disparity saved. File exists: {os.path.exists(dist_benign_raw_path)}, size={os.path.getsize(dist_benign_raw_path) if os.path.exists(dist_benign_raw_path) else 'N/A'} bytes")
                                else:
                                    # print(f"[DEBUG] Skipping distorted-benign raw disparity save (flag=False)")
                                    pass
                                
                                if args.save_disparity_maps:
                                    # print(f"[DEBUG] Preparing distorted-benign disparity map image")
                                    dm_dist_benign = np.array(disp_unp, dtype=np.float32)
                                    finite_mask_dist_benign = np.isfinite(dm_dist_benign)
                                    finite_count_dist_benign = np.sum(finite_mask_dist_benign)
                                    total_count_dist_benign = dm_dist_benign.size
                                    # print(f"[DEBUG] Distorted-benign finite values: {finite_count_dist_benign}/{total_count_dist_benign} ({100*finite_count_dist_benign/total_count_dist_benign:.2f}%)")
                                    
                                    if not np.any(finite_mask_dist_benign):
                                        # print(f"[DEBUG] No finite values in distorted-benign, using zeros")
                                        disp_norm_dist_benign = np.zeros_like(dm_dist_benign, dtype=np.uint8)
                                    else:
                                        dmin_dist_benign = float(np.nanmin(dm_dist_benign))
                                        dmax_dist_benign = float(np.nanmax(dm_dist_benign))
                                        denom_dist_benign = max(dmax_dist_benign - dmin_dist_benign, 1e-8)
                                        # print(f"[DEBUG] Normalizing distorted-benign: dmin={dmin_dist_benign:.3f}, dmax={dmax_dist_benign:.3f}, denom={denom_dist_benign:.3f}")
                                        disp_norm_dist_benign = ((dm_dist_benign - dmin_dist_benign) / denom_dist_benign * 255.0).astype(np.uint8)
                                        # print(f"[DEBUG] Distorted-benign normalized image stats: min={disp_norm_dist_benign.min()}, max={disp_norm_dist_benign.max()}, mean={disp_norm_dist_benign.mean():.2f}")
                                    
                                    dist_benign_png_path = os.path.join(dist_benign_disp_dir, f"{stem}_distorted_benign.png")
                                    # print(f"[DEBUG] Saving distorted-benign disparity map image to: {dist_benign_png_path}")
                                    # print(f"[DEBUG] Distorted-benign image shape: {disp_norm_dist_benign.shape}, dtype: {disp_norm_dist_benign.dtype}")
                                    Image.fromarray(disp_norm_dist_benign).save(dist_benign_png_path)
                                    # print(f"[DEBUG] Distorted-benign disparity map saved. File exists: {os.path.exists(dist_benign_png_path)}, size={os.path.getsize(dist_benign_png_path) if os.path.exists(dist_benign_png_path) else 'N/A'} bytes")
                                else:
                                    # print(f"[DEBUG] Skipping distorted-benign disparity map save (flag=False)")
                                    pass
                                
                                # print(f"[DEBUG] Distorted-benign disparity map saving completed successfully for {fname}")
                            except Exception as e:
                                import traceback
                                print(f"[ERROR] Failed to save distorted-benign disparity maps for {fname}: {e}")
                                print(f"[ERROR] Traceback:")
                                traceback.print_exc()

                        # End timing after all distorted-benign processing
                        dt_dist_benign = perf_counter() - t_dist_benign_start
                        if args.eta_include_benign:
                            pt.update(dt_dist_benign)
                            eta_db = pt.eta()
                            if eta_db is not None:
                                remain_db, eta_td_db = eta_db
                                print(f"[timing] distorted-benign pair={fname} dt={dt_dist_benign*1000:.2f} ms | "
                                      f"avg={pt.avg_dt*1000:.2f} ms | remaining={remain_db} | ETA={eta_td_db}")
                            else:
                                print(f"[timing] distorted-benign pair={fname} dt={dt_dist_benign*1000:.2f} ms | "
                                      f"avg=warming | remaining={pt.total - pt.count}")

                else:
                    distorted_benign_stats_cache = None

            for pat in pattern_items:
                freq = int(pat['freq'])
                gran = pat.get('granularity', None)
                stereo_item = pat.get('stereo', False)
                external_item = pat.get('external', False)
                pathL = pat.get('pathL', None); pathR = pat.get('pathR', None)
                shared_path = pat.get('path', None)

                pilL = load_pil(pathL) if (stereo_item and external_item and pathL) else None
                pilR = load_pil(pathR) if (stereo_item and external_item and pathR) else None
                pilS = load_pil(shared_path) if (external_item and shared_path and not stereo_item) else None

                print(f"[pattern] ratio={ratio:.3f} freq={freq} gran={gran} stereo={stereo_item} external={external_item} calib={calib_mag}")

                for disp in disp_list:
                    pat_key = (float(ratio), int(freq), gran, int(disp), int(calib_mag))
                    combo_key = (ratio, freq, gran, disp, calib_mag)
                    done_pairs = done_pairs_per_combo.get(combo_key, set())

                    if len(done_pairs) >= len(pairs):
                        skipped_pattern_combos += 1
                        print(f" [skip] pattern fully complete for ratio={ratio:.3f}, "
                              f"freq={freq}, gran={gran}, disp={disp}, calib={calib_mag} "
                              f"({len(done_pairs)}/{len(pairs)} pairs); skipping inner loop")
                        continue

                    print(f" [disp] shift={disp} | done_pairs={len(done_pairs)}/{len(pairs)} | calib={calib_mag}")

                    for idx, fname in enumerate(pairs):
                        row_key = (fname, ratio, freq, gran, disp, calib_mag)
                        if row_key in done_rows:
                            # print(f"[DEBUG] Skipping {fname} - already in done_rows")
                            continue
                        # print(f"[DEBUG] Proceeding with {fname} - not in done_rows")

                        # Start timing from the beginning of unit processing (includes I/O, prep, inference, stats, CSV)
                        t_unit_start = perf_counter()

                        left_path = os.path.join(args.left_dir, fname)
                        right_path = os.path.join(args.right_dir, fname)
                        try:
                            limg = np.array(Image.open(left_path).convert("RGB"))
                            rimg = np.array(Image.open(right_path).convert("RGB"))
                            # print(f"[DEBUG] Images loaded: limg shape={limg.shape}, rimg shape={rimg.shape}")
                        except Exception as e:
                            print(f"[ERROR] Skipping {fname}: failed to load images ({e})")
                            import traceback
                            traceback.print_exc()
                            continue

                        stem, _ext = os.path.splitext(fname)
                        h_img, w_img = limg.shape[:2]
                        cx, cy = w_img // 2, h_img // 2
                        # print(f"[DEBUG] Image dimensions: {h_img}x{w_img}, center=({cx}, {cy}), stem={stem}")

                        use_full_pattern = args.full_pattern and external_item
                        # print(f"[DEBUG] Pattern generation: use_full_pattern={use_full_pattern}, external_item={external_item}, stereo_item={stereo_item}")
                        if use_full_pattern:
                            # print(f"[DEBUG] Using full pattern mode")
                            pass
                            if stereo_item and pathL:
                                base_img = Image.open(pathL).convert('RGB')
                                pat_full = np.array(base_img, dtype=np.uint8)
                                ph_full, pw_full = pat_full.shape[:2]
                                max_ph = int(0.75 * h_img)
                                if ph_full > max_ph:
                                    crop_top = (ph_full - max_ph) // 2
                                    crop_bottom = crop_top + max_ph
                                    pat_full = pat_full[crop_top:crop_bottom, :, :]
                                    ph_full = pat_full.shape[0]
                                pattern_left = pat_full

                                if pathR:
                                    base_img_r = Image.open(pathR).convert('RGB')
                                    pat_r_full = np.array(base_img_r, dtype=np.uint8)
                                    ph_r, pw_r = pat_r_full.shape[:2]
                                    max_ph_r = int(0.75 * h_img)
                                    if ph_r > max_ph_r:
                                        crop_top_r = (ph_r - max_ph_r) // 2
                                        crop_bottom_r = crop_top_r + max_ph_r
                                        pat_r_full = pat_r_full[crop_top_r:crop_bottom_r, :, :]
                                    pattern_right = pat_r_full
                                else:
                                    pattern_right = pattern_left.copy()
                            elif shared_path:
                                base_img = Image.open(shared_path).convert('RGB')
                                pat_full = np.array(base_img, dtype=np.uint8)
                                ph_full, pw_full = pat_full.shape[:2]
                                max_ph = int(0.75 * h_img)
                                if ph_full > max_ph:
                                    crop_top = (ph_full - max_ph) // 2
                                    crop_bottom = crop_top + max_ph
                                    pat_full = pat_full[crop_top:crop_bottom, :, :]
                                    ph_full = pat_full.shape[0]
                                pattern_left = pat_full
                                pattern_right = pattern_left.copy()
                            else:
                                # Fall through to normal pattern generation
                                use_full_pattern = False

                            if use_full_pattern and args.luminance != 'none':
                                # Apply luminance adjustment to full pattern
                                pat_luma = rgb_to_luma_uint8(pattern_left).astype(np.float32)
                                bg_luma_full = rgb_to_luma_uint8(limg).astype(np.float32)
                                if args.luminance == 'local':
                                    bg_luma_blur = cv2.GaussianBlur(bg_luma_full, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REPLICATE)
                                    bg_luma_mean = cv2.resize(bg_luma_blur, (pattern_left.shape[1], pattern_left.shape[0]), interpolation=cv2.INTER_NEAREST)
                                else:
                                    avg = float(bg_luma_full.mean())
                                    bg_luma_mean = np.full((pattern_left.shape[0], pattern_left.shape[1]), avg, dtype=np.float32)
                                pmin, pmax = float(np.min(pat_luma)), float(np.max(pat_luma))
                                if pmax - pmin < 1e-6:
                                    norm = np.zeros_like(pat_luma, dtype=np.float32)
                                else:
                                    norm = (pat_luma - pmin) / (pmax - pmin)
                                amplitude = 127.0 * float(np.clip(args.luminance_contrast, 0.0, 1.0))
                                out_luma = np.clip(bg_luma_mean + (norm - 0.5) * 2.0 * amplitude, 0, 255).astype(np.uint8)
                                pattern_left = np.repeat(out_luma[..., None], 3, axis=-1)
                                if stereo_item and pathR:
                                    # Similar for right
                                    pat_r_luma = rgb_to_luma_uint8(pattern_right).astype(np.float32)
                                    bg_r_luma_full = rgb_to_luma_uint8(rimg).astype(np.float32)
                                    if args.luminance == 'local':
                                        bg_r_luma_blur = cv2.GaussianBlur(bg_r_luma_full, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REPLICATE)
                                        bg_r_luma_mean = cv2.resize(bg_r_luma_blur, (pattern_right.shape[1], pattern_right.shape[0]), interpolation=cv2.INTER_NEAREST)
                                    else:
                                        avg_r = float(bg_r_luma_full.mean())
                                        bg_r_luma_mean = np.full((pattern_right.shape[0], pattern_right.shape[1]), avg_r, dtype=np.float32)
                                    pmin_r, pmax_r = float(np.min(pat_r_luma)), float(np.max(pat_r_luma))
                                    if pmax_r - pmin_r < 1e-6:
                                        norm_r = np.zeros_like(pat_r_luma, dtype=np.float32)
                                    else:
                                        norm_r = (pat_r_luma - pmin_r) / (pmax_r - pmin_r)
                                    out_r_luma = np.clip(bg_r_luma_mean + (norm_r - 0.5) * 2.0 * amplitude, 0, 255).astype(np.uint8)
                                    pattern_right = np.repeat(out_r_luma[..., None], 3, axis=-1)
                                else:
                                    pattern_right = pattern_left.copy()
                            else:
                                # Normal pattern generation (when using full pattern but luminance is 'none')
                                ph = int(math.ceil(h_img * ratio))
                                pw = int(math.ceil(w_img * ratio))
                                if stereo_item and pilL is not None and pilR is not None:
                                    # print(f"[DEBUG] Using stereo external patterns")
                                    pass
                                    pattern_left = prepare_pattern_from_image_cached(
                                        pilL, ph, pw, limg, args.luminance, args.luminance_contrast, keep_aspect=args.keep_aspect)
                                    pattern_right = prepare_pattern_from_image_cached(
                                        pilR, ph, pw, rimg, args.luminance, args.luminance_contrast, keep_aspect=args.keep_aspect)
                                elif external_item and pilS is not None:
                                    pattern_left = prepare_pattern_from_image_cached(
                                        pilS, ph, pw, limg, args.luminance, args.luminance_contrast, keep_aspect=args.keep_aspect)
                                    pattern_right = prepare_pattern_from_image_cached(
                                        pilS, ph, pw, rimg, args.luminance, args.luminance_contrast, keep_aspect=args.keep_aspect)
                                else:
                                    if args.luminance == "none":
                                        pattern_left = generate_stripes(ph, pw, freq)
                                        pattern_right = pattern_left
                                    else:
                                        pattern_left = generate_stripes_luminance(
                                            ph, pw, freq, limg, mode=args.luminance, contrast=args.luminance_contrast)
                                        pattern_right = generate_stripes_luminance(
                                            ph, pw, freq, rimg, mode=args.luminance, contrast=args.luminance_contrast)
                        else:
                            # Normal pattern generation (when NOT using full pattern)
                            ph = int(math.ceil(h_img * ratio))
                            pw = int(math.ceil(w_img * ratio))
                            # print(f"[DEBUG] Pattern size: ph={ph}, pw={pw} (ratio={ratio})")
                            # print(f"[DEBUG] Pattern sources: stereo_item={stereo_item}, pilL={pilL is not None}, pilR={pilR is not None}, pilS={pilS is not None}, external_item={external_item}")
                            if stereo_item and pilL is not None and pilR is not None:
                                # print(f"[DEBUG] Using stereo external patterns")
                                pass
                                pattern_left = prepare_pattern_from_image_cached(
                                    pilL, ph, pw, limg, args.luminance, args.luminance_contrast, keep_aspect=args.keep_aspect)
                                pattern_right = prepare_pattern_from_image_cached(
                                    pilR, ph, pw, rimg, args.luminance, args.luminance_contrast, keep_aspect=args.keep_aspect)
                            elif external_item and pilS is not None:
                                pattern_left = prepare_pattern_from_image_cached(
                                    pilS, ph, pw, limg, args.luminance, args.luminance_contrast, keep_aspect=args.keep_aspect)
                                pattern_right = prepare_pattern_from_image_cached(
                                    pilS, ph, pw, rimg, args.luminance, args.luminance_contrast, keep_aspect=args.keep_aspect)
                            else:
                                if args.luminance == "none":
                                    pattern_left = generate_stripes(ph, pw, freq)
                                    pattern_right = pattern_left
                                else:
                                    pattern_left = generate_stripes_luminance(
                                        ph, pw, freq, limg, mode=args.luminance, contrast=args.luminance_contrast)
                                    pattern_right = generate_stripes_luminance(
                                        ph, pw, freq, rimg, mode=args.luminance, contrast=args.luminance_contrast)

                        l_atk = place_pattern_on_image(limg, pattern_left, cx, cy)
                        r_atk = place_pattern_on_image(rimg, pattern_right, cx - disp, cy)
                        # print(f"[DEBUG] Patterns placed: l_atk shape={l_atk.shape}, r_atk shape={r_atk.shape}")

                        if use_calib: # Unused presently.
                            # print(f"[DEBUG] Applying calibration error distortion")
                            pass
                            l_atk = apply_distortion_to_ideal_image(l_atk, K_L_err, D_L_err)
                            r_atk = apply_distortion_to_ideal_image(r_atk, K_R_err, D_R_err)

                        # Save synthesized attack images (with patterns placed) right before inference
                        if args.save_raw_disparity or args.save_disparity_maps:
                            try:
                                gran_str = f"_g{gran}" if gran is not None else ""
                                # Save to input_images/attack subdirectory
                                atk_img_subdir = os.path.join(attack_images_dir, f"{stem}_f{freq}{gran_str}_d{disp}_ratio{ratio:.3f}_calibM{calib_mag}")
                                os.makedirs(atk_img_subdir, exist_ok=True)
                                
                                # Save left synthesized image
                                l_atk_img_path = os.path.join(atk_img_subdir, f"{stem}_left.png")
                                # print(f"[DEBUG] Saving left synthesized attack image to: {l_atk_img_path}")
                                Image.fromarray(l_atk).save(l_atk_img_path)
                                # print(f"[DEBUG] Left synthesized attack image saved. File exists: {os.path.exists(l_atk_img_path)}, size={os.path.getsize(l_atk_img_path) if os.path.exists(l_atk_img_path) else 'N/A'} bytes")
                                
                                # Save right synthesized image
                                r_atk_img_path = os.path.join(atk_img_subdir, f"{stem}_right.png")
                                # print(f"[DEBUG] Saving right synthesized attack image to: {r_atk_img_path}")
                                Image.fromarray(r_atk).save(r_atk_img_path)
                                # print(f"[DEBUG] Right synthesized attack image saved. File exists: {os.path.exists(r_atk_img_path)}, size={os.path.getsize(r_atk_img_path) if os.path.exists(r_atk_img_path) else 'N/A'} bytes")
                            except Exception as e:
                                import traceback
                                print(f"[ERROR] Failed to save synthesized attack images for {fname}: {e}")
                                print(f"[ERROR] Traceback:")
                                traceback.print_exc()

                        # print(f"[DEBUG] Running UniMatch inference for {fname}")
                        atk_disp = run_unimatch_np(
                            model, l_atk, r_atk, device,
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            attn_type=args.attn_type,
                            attn_splits_list=args.attn_splits_list,
                            corr_radius_list=args.corr_radius_list,
                            prop_radius_list=args.prop_radius_list,
                            num_reg_refine=args.num_reg_refine,
                            use_amp=args.amp
                        )
                        # print(f"[DEBUG] Inference complete: atk_disp shape={atk_disp.shape}, dtype={atk_disp.dtype}")

                        mean_ben, max_ben, min_ben, var_ben, std_ben, med_ben = benign_stats_cache[fname]
                        mean_ben_in, max_ben_in, min_ben_in, var_ben_in, std_ben_in, med_ben_in = benign_inner_stats_cache[fname]

                        cy, cx, ph, pw, inner_ph, inner_pw = center_region_meta[fname]
                        mean_atk, max_atk, min_atk, var_atk, std_atk, med_atk = get_center_region_stats(
                            atk_disp, cy, cx, ph, pw
                        )

                        mean_atk_in, max_atk_in, min_atk_in, var_atk_in, std_atk_in, med_atk_in = \
                            get_center_region_stats(atk_disp, cy, cx, inner_ph, inner_pw)

                        # D1 scores: using fixed discrepancy (disp) as ground truth
                        # Full patch region
                        d1_full = compute_d1_score(atk_disp, cy, cx, ph, pw, ground_truth_disp=disp, threshold=3.0)
                        # Inner patch region
                        d1_inner = compute_d1_score(atk_disp, cy, cx, inner_ph, inner_pw, ground_truth_disp=disp, threshold=3.0)

                        # attack histogram over full patch region
                        hist = center_region_histogram(atk_disp, cy, cx, ph, pw, max_disp=400)
                        hist_row = {
                            'pair': fname,
                            'ratio': ratio,
                            'frequency': freq,
                            'granularity': gran,
                            'discrepency': disp,
                            'calib_magnitude': calib_mag,
                            'phase': 'attack',
                        }
                        for d in range(401):
                            hist_row[f'd{d}'] = int(hist[d])
                        k_full = (fname, float(ratio), int(freq), _norm_hist_gran(gran), int(disp), int(calib_mag), 'attack')
                        if k_full not in done_hist_full:
                            hist_writer_full.writerow(hist_row)
                            hist_full_f.flush()
                            done_hist_full.add(k_full)

                        # attack inner-center histogram
                        hist_inner = center_region_histogram(atk_disp, cy, cx, inner_ph, inner_pw, max_disp=400)
                        hist_inner_row = {
                            'pair': fname,
                            'ratio': ratio,
                            'frequency': freq,
                            'granularity': gran,
                            'discrepency': disp,
                            'calib_magnitude': calib_mag,
                            'phase': 'attack',
                        }
                        for d in range(401):
                            hist_inner_row[f'd{d}'] = int(hist_inner[d])
                        k_inner = (fname, float(ratio), int(freq), _norm_hist_gran(gran), int(disp), int(calib_mag), 'attack')
                        if k_inner not in done_hist_inner:
                            hist_writer_inner.writerow(hist_inner_row)
                            hist_inner_f.flush()
                            done_hist_inner.add(k_inner)

                        try:
                            if hasattr(atk_disp, 'shape'):
                                pass
                            if hasattr(atk_disp, 'dtype'):
                                pass
                        except NameError:
                            pass
                        
                        # Save attack disparity maps if requested
                        if args.save_raw_disparity or args.save_disparity_maps:
                            
                            try:
                                gran_str = f"_g{gran}" if gran is not None else ""
                                # Save to disparity_maps/attack subdirectory
                                atk_disp_dir = os.path.join(disparity_maps_root, "attack", f"{stem}_f{freq}{gran_str}_d{disp}_ratio{ratio:.3f}_calibM{calib_mag}")
                                # print(f"[DEBUG] Creating attack disparity directory: {atk_disp_dir}")
                                os.makedirs(atk_disp_dir, exist_ok=True)
                                # print(f"[DEBUG] Attack disparity directory created/exists: {os.path.exists(atk_disp_dir)}")
                                
                                if args.save_raw_disparity:
                                    raw_file_path = os.path.join(atk_disp_dir, f"{stem}_attack.npy")
                                    np.save(raw_file_path, atk_disp)
                                else:
                                    pass
                                
                                if args.save_disparity_maps:
                                    # print(f"[DEBUG] Preparing attack disparity map image")
                                    dm = np.array(atk_disp, dtype=np.float32)
                                    finite_mask = np.isfinite(dm)
                                    finite_count = np.sum(finite_mask)
                                    total_count = dm.size
                                    # print(f"[DEBUG] Attack finite values: {finite_count}/{total_count} ({100*finite_count/total_count:.2f}%)")
                                    
                                    if not np.any(finite_mask):
                                        # print(f"[DEBUG] No finite values, using zeros")
                                        disp_norm = np.zeros_like(dm, dtype=np.uint8)
                                    else:
                                        dmin = float(np.nanmin(dm))
                                        dmax = float(np.nanmax(dm))
                                        denom = max(dmax - dmin, 1e-8)
                                        # print(f"[DEBUG] Normalizing attack: dmin={dmin:.3f}, dmax={dmax:.3f}, denom={denom:.3f}")
                                        disp_norm = ((dm - dmin) / denom * 255.0).astype(np.uint8)
                                        # print(f"[DEBUG] Attack normalized image stats: min={disp_norm.min()}, max={disp_norm.max()}, mean={disp_norm.mean():.2f}")
                                    
                                    png_file_path = os.path.join(atk_disp_dir, f"{stem}_attack.png")
                                    Image.fromarray(disp_norm).save(png_file_path)
                                    # print(f"[DEBUG] Attack disparity map saved. File exists: {os.path.exists(png_file_path)}, size={os.path.getsize(png_file_path) if os.path.exists(png_file_path) else 'N/A'} bytes")
                                else:
                                    # print(f"[DEBUG] Skipping attack disparity map save (flag=False)")
                                    pass
                                
                                # print(f"[DEBUG] Attack disparity map saving completed successfully for {fname}")
                            except Exception as e:
                                import traceback
                                print(f"[ERROR] Failed to save attack disparity maps for {fname}: {e}")
                                print(f"[ERROR] Traceback:")
                                traceback.print_exc()
                        else:
                            # print(f"[DEBUG] Skipping attack disparity map save (both flags are False)")
                            pass

                        row = {
                            'pair': fname,
                            'ratio': ratio,
                            'frequency': freq,
                            'granularity': gran,
                            'discrepency': disp,
                            'calib_magnitude': calib_mag,
                            'mean_benign': mean_ben,
                            'mean_attack': mean_atk,
                            'mean_diff': mean_atk - mean_ben,
                            'max_benign': max_ben,
                            'max_attack': max_atk,
                            'max_diff': max_atk - max_ben,
                            'min_benign': min_ben,
                            'min_attack': min_atk,
                            'min_diff': min_atk - min_ben,
                            'var_benign': var_ben,
                            'var_attack': var_atk,
                            'var_diff': var_atk - var_ben,
                            'std_benign': std_ben,
                            'std_attack': std_atk,
                            'std_diff': std_atk - std_ben,
                            'median_benign': med_ben,
                            'median_attack': med_atk,
                            'median_diff': med_atk - med_ben,
                            # Inner-center stats
                            'mean_benign_inner': mean_ben_in,
                            'mean_attack_inner': mean_atk_in,
                            'mean_diff_inner': mean_atk_in - mean_ben_in,
                            'max_benign_inner': max_ben_in,
                            'max_attack_inner': max_atk_in,
                            'max_diff_inner': max_atk_in - max_ben_in,
                            'min_benign_inner': min_ben_in,
                            'min_attack_inner': min_atk_in,
                            'min_diff_inner': min_atk_in - min_ben_in,
                            'var_benign_inner': var_ben_in,
                            'var_attack_inner': var_atk_in,
                            'var_diff_inner': var_atk_in - var_ben_in,
                            'std_benign_inner': std_ben_in,
                            'std_attack_inner': std_atk_in,
                            'std_diff_inner': std_atk_in - std_ben_in,
                            'median_benign_inner': med_ben_in,
                            'median_attack_inner': med_atk_in,
                            'median_diff_inner': med_atk_in - med_ben_in,
                            # D1 scores (using fixed discrepancy as ground truth)
                            'd1_full': d1_full,
                            'd1_inner': d1_inner,
                        }

                        if distorted_benign_stats_cache is not None and fname in distorted_benign_stats_cache:
                            mean_bd, max_bd, min_bd, var_bd, std_bd, med_bd = distorted_benign_stats_cache[fname]
                            row['mean_benign_dist'] = mean_bd
                            row['max_benign_dist'] = max_bd
                            row['min_benign_dist'] = min_bd
                            row['var_benign_dist'] = var_bd
                            row['std_benign_dist'] = std_bd
                            row['median_benign_dist'] = med_bd
                        else:
                            row['mean_benign_dist'] = ''
                            row['max_benign_dist'] = ''
                            row['min_benign_dist'] = ''
                            row['var_benign_dist'] = ''
                            row['std_benign_dist'] = ''
                            row['median_benign_dist'] = ''

                        # Defensive: if CSV couldn't be opened/writer couldn't be created, fail loudly.
                        if w is None:
                            raise RuntimeError("CSV writer is not initialized; cannot write results row")
                        w.writerow(row)
                        rows_since_flush += 1

                        done_rows.add(row_key)
                        done_pairs_per_combo[combo_key].add(fname)

                        if rows_since_flush >= args.flush_every:
                            csv_f.flush()
                            rows_since_flush = 0

                        # End timing after all processing (I/O, prep, inference, stats, CSV writing)
                        dt_unit = perf_counter() - t_unit_start
                        pt.update(dt_unit)
                        eta_info = pt.eta()
                        if eta_info is not None:
                            remain, eta_td = eta_info
                            print(f"[timing] attack pair={fname} dt={dt_unit*1000:.2f} ms | "
                                  f"avg={pt.avg_dt*1000:.2f} ms | remaining={remain} | ETA={eta_td}")
                        else:
                            print(f"[timing] attack pair={fname} dt={dt_unit*1000:.2f} ms | "
                                  f"avg=warming | remaining={pt.total - pt.count}")

        if rows_since_flush > 0:
            csv_f.flush()

        if pt.count:
            print(f"[progress] completed {pt.count}/{pt.total} inferences "
                  f"(skipped pattern-combos={skipped_pattern_combos}) | "
                  f"avg={pt.avg_dt*1000:.2f} ms")
        else:
            print("[progress] no inferences executed")

    finally:
        # Close only if successfully opened.
        try:
            if csv_f is not None:
                csv_f.close()
        finally:
            # Try to close histogram files even if CSV close fails.
            if hist_full_f is not None:
                hist_full_f.close()
            if hist_inner_f is not None:
                hist_inner_f.close()
        print(f"\nResults streaming to {csv_path}, {hist_full_csv_path} and {hist_inner_csv_path} (closed at end)")


if __name__ == "__main__":
    main()
