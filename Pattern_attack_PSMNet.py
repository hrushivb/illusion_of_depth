import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

import numpy as np
import argparse
import math
import os
import cv2
import re
from glob import glob
import csv
import contextlib
import time

from models import *

torch.backends.cudnn.benchmark = True

# Fraction of the patch size to use to calculate the center region statistics in order to avoid edge artifacts
INNER_CENTER_RATIO = 0.8

# Convert RGB image to luma
def rgb_to_luma_uint8(img_rgb_uint8):
    r = img_rgb_uint8[..., 0].astype(np.float32)
    g = img_rgb_uint8[..., 1].astype(np.float32)
    b = img_rgb_uint8[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)


# Generate a pattern of stripes
def generate_stripes(height, width, frequency):
    pattern = np.zeros((height, width), dtype=np.uint8)
    stripe_w = max(1, int(frequency))
    for i in range(0, width, 2 * stripe_w):
        pattern[:, i:i + stripe_w] = 255
    return np.stack([pattern] * 3, axis=-1)


# Generate a pattern of stripes with luminance
def generate_stripes_luminance(height, width, frequency, full_bg, mode='mean', contrast=0.6):
    stripe_mask = np.zeros((height, width), dtype=np.uint8)
    stripe_w = max(1, int(frequency))
    for i in range(0, width, 2 * stripe_w):
        stripe_mask[:, i:i + stripe_w] = 255

    if mode == 'none':
        pattern = np.stack([stripe_mask] * 3, axis=-1)
        return pattern

    full_luma = rgb_to_luma_uint8(full_bg).astype(np.float32)

    if mode == 'local':
        luma_sm = cv2.GaussianBlur(full_luma, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_REPLICATE)
        global_mean = float(luma_sm.mean())
    else:
        global_mean = float(full_luma.mean())

    luma_mean = np.full((height, width), global_mean, dtype=np.float32)
    amplitude = 127.0 * float(np.clip(contrast, 0.0, 1.0))
    bright = np.clip(luma_mean + amplitude, 0, 255)
    dark = np.clip(luma_mean - amplitude, 0, 255)
    pattern_luma = np.where(stripe_mask == 255, bright, dark).astype(np.uint8)
    return np.repeat(pattern_luma[..., None], 3, axis=-1)


# Place a pattern on an image
def place_pattern_on_image(image, pattern, center_x, center_y):
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


# Forward pass for a batch of images
def forward_batched(model, imgL_batch, imgR_batch, device, use_amp=True):
    model.eval()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device.type == "cuda") else contextlib.nullcontext()
    with torch.inference_mode():
        with amp_ctx:
            disp = model(imgL_batch.to(device, non_blocking=True),
                         imgR_batch.to(device, non_blocking=True))

    if disp.dim() == 4 and disp.size(1) == 1:
        disp = disp[:, 0, :, :]
    elif disp.dim() == 3:
        pass
    else:
        raise ValueError(f"Unexpected disp shape {tuple(disp.shape)}; expected (N,1,H,W)")
    return disp.detach().cpu().numpy()

# Convert a numpy array to a tensor and add padding
def to_tensor_normalized_padded(np_img, infer_transform, pad_mult=16):
    img = Image.fromarray(np_img)
    t = infer_transform(img)
    H, W = t.shape[1], t.shape[2]
    top_pad = (pad_mult - H % pad_mult) if H % pad_mult != 0 else 0
    right_pad = (pad_mult - W % pad_mult) if W % pad_mult != 0 else 0
    t_pad = F.pad(t, (0, right_pad, top_pad, 0))
    return t_pad, top_pad, right_pad


# Unpad a disparity map
def unpad_disparity(np_disp, top_pad, right_pad):
    if np_disp.ndim == 3:
        if top_pad and right_pad:
            return np_disp[:, top_pad:, :-right_pad]
        elif right_pad:
            return np_disp[:, :, :-right_pad]
        elif top_pad:
            return np_disp[:, top_pad:, :]
        else:
            return np_disp
    elif np_disp.ndim == 2:
        if top_pad and right_pad:
            return np_disp[top_pad:, :-right_pad]
        elif right_pad:
            return np_disp[:, :-right_pad]
        elif top_pad:
            return np_disp[top_pad:, :]
        else:
            return np_disp
    else:
        raise ValueError(f"Unexpected disparity ndim={np_disp.ndim}, expected 2 or 3")

# Extract a center region from a disparity map
def _extract_center_region(disp_map, center_y, center_x, region_h, region_w):

    h, w = disp_map.shape

    # Compute top-left corner so that the region is centered at (center_y, center_x)
    y1 = max(center_y - region_h // 2, 0)
    x1 = max(center_x - region_w // 2, 0)

    # Clamp bottom-right corner to image bounds
    y2 = min(y1 + region_h, h)
    x2 = min(x1 + region_w, w)

    if y1 >= y2 or x1 >= x2:
        return disp_map

    return disp_map[y1:y2, x1:x2]


def get_center_region_stats(disp_map, center_y, center_x, region_h, region_w):
    center_region = _extract_center_region(disp_map, center_y, center_x, region_h, region_w)
    avg_disp = center_region.mean()
    max_disp = center_region.max()
    min_disp = center_region.min()
    var_disp = center_region.var()
    std_disp = center_region.std()
    median_disp = np.median(center_region)
    return avg_disp, max_disp, min_disp, var_disp, std_disp, median_disp


def center_region_histogram(disp_map, center_y, center_x, region_h, region_w, max_disp=192):
    center = _extract_center_region(disp_map, center_y, center_x, region_h, region_w)

    disp_int = np.rint(center).astype(np.int32)
    disp_int = np.clip(disp_int, 0, max_disp)
    hist = np.bincount(disp_int.ravel(), minlength=max_disp + 1)
    return hist


# Unused presently
def compute_d1_score(disp_map, center_y, center_x, region_h, region_w, ground_truth_disp, threshold=3.0):
    center_region = _extract_center_region(disp_map, center_y, center_x, region_h, region_w)
    
    # Compute absolute error: |predicted - ground_truth|
    error = np.abs(center_region - ground_truth_disp)
    
    # Count pixels with error > threshold
    total_pixels = center_region.size
    bad_pixels = int((error > threshold).sum())
    
    d1_score = bad_pixels / float(max(1, total_pixels))
    return d1_score


def parse_frequency_from_dirname(dirname: str):
    m = re.match(r'^size_(\d+)$', os.path.basename(dirname))
    return int(m.group(1)) if m else None


def parse_granularity_from_filename(path: str):
    base = os.path.splitext(os.path.basename(path))[0]
    return base


def load_external_patterns(pattern_dir: str):
    patterns = []
    if not pattern_dir or not os.path.isdir(pattern_dir):
        return patterns

    size_dirs = [d for d in glob(os.path.join(pattern_dir, 'size_*')) if os.path.isdir(d)]
    for sd in sorted(size_dirs):
        freq = parse_frequency_from_dirname(sd)
        if freq is None:
            continue

        img_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
            img_paths.extend(glob(os.path.join(sd, ext)))

        for ip in sorted(img_paths):
            gran = parse_granularity_from_filename(ip)
            patterns.append({'freq': int(freq), 'granularity': gran, 'path': ip})

    return patterns


def prepare_pattern_from_image(pat_img_path: str, ph: int, pw: int, full_bg: np.ndarray,
                               luminance_mode: str, luminance_contrast: float,
                               keep_aspect: bool = False) -> np.ndarray:
    img = Image.open(pat_img_path).convert('RGB')
    orig_w, orig_h = img.size

    if keep_aspect:
        aspect = orig_w / float(orig_h)
        pw = int(round(ph * aspect))

    img = img.resize((pw, ph), resample=Image.NEAREST)
    pat = np.array(img, dtype=np.uint8)

    if luminance_mode == 'none':
        return pat

    pat_luma = rgb_to_luma_uint8(pat).astype(np.float32)
    bg_luma_full = rgb_to_luma_uint8(full_bg).astype(np.float32)

    if luminance_mode == 'local':
        bg_luma_blur = cv2.GaussianBlur(bg_luma_full, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REPLICATE)
        bg_luma_mean = cv2.resize(bg_luma_blur, (pw, ph), interpolation=cv2.INTER_NEAREST)
    else:
        avg = float(bg_luma_full.mean())
        bg_luma_mean = np.full((ph, pw), avg, dtype=np.float32)

    pmin, pmax = float(np.min(pat_luma)), float(np.max(pat_luma))
    if pmax - pmin < 1e-6:
        norm = np.zeros_like(pat_luma, dtype=np.float32)
    else:
        norm = (pat_luma - pmin) / (pmax - pmin)

    amplitude = 127.0 * float(np.clip(luminance_contrast, 0.0, 1.0))
    out_luma = np.clip(bg_luma_mean + (norm - 0.5) * 2.0 * amplitude, 0, 255).astype(np.uint8)
    out = np.repeat(out_luma[..., None], 3, axis=-1)
    return out


def _natural_sort_key(s):
    parts = re.split(r'(\d+)', os.path.basename(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _find_pattern_image_legacy_fg(directory, freq, granularity=None):
    if not directory or not os.path.isdir(directory):
        return None

    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    candidates = []

    if granularity is not None:
        for ext in exts:
            candidates.extend(glob(os.path.join(directory, f"*_{freq}_{granularity}{ext[1:]}")))
        if not candidates:
            for ext in exts:
                candidates.extend(glob(os.path.join(directory, f"*_{freq}_{granularity}*")))
        if candidates:
            candidates.sort(key=_natural_sort_key)
            return candidates[0]

    candidates = []
    for ext in exts:
        candidates.extend(glob(os.path.join(directory, f"*_{freq}{ext[1:]}")))
    if not candidates:
        for ext in exts:
            candidates.extend(glob(os.path.join(directory, f"*_{freq}*")))
    if candidates:
        candidates.sort(key=_natural_sort_key)
        return candidates[0]

    return None


def _fallback_generate(ph, pw, freq, full_bg, luminance_mode, luminance_contrast):
    if luminance_mode == 'none':
        return generate_stripes(ph, pw, freq)
    return generate_stripes_luminance(ph, pw, freq, full_bg, mode=luminance_mode, contrast=luminance_contrast)


def pad_to_target(t, target_h, target_w):
    h, w = t.shape[1], t.shape[2]
    pad_top = target_h - h
    pad_right = target_w - w
    if pad_top > 0 or pad_right > 0:
        t = F.pad(t, (0, pad_right, pad_top, 0))
    return t


def natural_sorted_files(dir_path):
    names = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    return sorted(names, key=_natural_sort_key)


def infer_array_env():
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    tcnt = os.environ.get("SLURM_ARRAY_TASK_COUNT") or os.environ.get("SLURM_TASKS_PER_NODE")

    try:
        task_id = int(tid) if tid is not None else 0
    except Exception:
        task_id = 0

    try:
        task_count = int(tcnt) if tcnt is not None else 1
    except Exception:
        task_count = 1

    task_id = max(0, task_id)
    task_count = max(1, task_count)

    if task_id >= task_count:
        task_id = task_id % task_count

    return task_id, task_count


def parse_index_range(s):
    if not s:
        return None
    m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', s)
    if not m:
        raise ValueError(f"--index_range must be a-b (inclusive), got: {s}")
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        raise ValueError(f"--index_range end < start: {s}")
    return a, b


def build_numeric_id_from_name(name: str):
    base = os.path.splitext(os.path.basename(name))[0]
    m = re.search(r'(\d+)', base)
    return int(m.group(1)) if m else None


def collect_pairs_by_numeric_id(left_dir, right_dir):
    left_names = natural_sorted_files(left_dir)
    right_names = set(natural_sorted_files(right_dir))

    common = [f for f in left_names if f in right_names]
    pairs = []
    for fname in common:
        nid = build_numeric_id_from_name(fname)
        if nid is not None:
            pairs.append((nid, fname))

    pairs.sort(key=lambda x: x[0])
    return pairs


def file_is_empty(path):
    try:
        return os.path.getsize(path) == 0
    except FileNotFoundError:
        return True


def fmt_hms(seconds):
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def load_progress(csv_path):
    done_patterns = set()
    done_pairs_by_pattern = {}
    if not os.path.exists(csv_path) or file_is_empty(csv_path):
        return done_patterns, done_pairs_by_pattern

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pair = row['pair']
                ratio = float(row['ratio'])
                freq = int(row['frequency'])
                gran = row['granularity']
                disp = int(row['discrepency'])
                calib_mag = int(row.get('calib_magnitude', 0))
                key = (ratio, freq, gran, disp, calib_mag)
                s = done_pairs_by_pattern.setdefault(key, set())
                s.add(pair)
            except Exception:
                continue

    return done_patterns, done_pairs_by_pattern


# -------- Calibration error helpers --------

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


def parse_fg_list(s: str):
    out = []
    if not s:
        return out
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' in tok:
            f_str, g_str = tok.split(':', 1)
            f = int(f_str.strip())
            g = g_str.strip()
            out.append((f, g))
        else:
            f = int(tok)
            out.append((f, None))
    return out

# NOTE: the notation used in the code does not match the notation used in the paper.
# The code uses the notation "Granularity" for the size of the black sqare within the checkerboard pattern, while the paper uses the notation "patch size".
# The code uses the notation "Frequency" for the frequency of the checkerboard pattern, while the paper uses the notation "Granularity".
def main():
    parser = argparse.ArgumentParser(description='Streaming-batch Stereo Disparity Evaluation')

    parser.add_argument('--left_dir', type=str, default="./KITTI Data/data_object_image_2/testing/image_2")
    parser.add_argument('--right_dir', type=str, default="./KITTI Data/data_object_image_3/testing/image_3")
    parser.add_argument('--ratios', type=str, default='0.25')
    parser.add_argument('--frequency', type=str, default='1,2,3,4,5')
    parser.add_argument('--discrepency', type=str, default='10')
    parser.add_argument('--max_pairs', type=int, default=10)
    parser.add_argument('--loadmodel', type=str, default='./saved_models/pretrained_model_KITTI2015.tar')
    parser.add_argument('--model', type=str, default='stackhourglass')
    parser.add_argument('--maxdisp', type=int, default=192)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--save_raw_disparity', action='store_true', default=False)
    parser.add_argument('--save_disparity_maps', action='store_true', default=False)
    parser.add_argument('--white_test', action='store_true', default=False)
    parser.add_argument('--luminance', type=str, default='none', choices=['none', 'mean', 'local'])
    parser.add_argument('--luminance_contrast', type=float, default=0.6)
    parser.add_argument('--attack_left_dir', type=str, default=None)
    parser.add_argument('--attack_right_dir', type=str, default=None)
    parser.add_argument('--pattern_dir', type=str, default='')
    parser.add_argument('--pattern_dir_left', type=str, default='')
    parser.add_argument('--pattern_dir_right', type=str, default='')
    parser.add_argument('--keep_aspect', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--index_range', type=str, default='')
    parser.add_argument('--chunk_size', type=int, default=100, help='Process images in chunks to bound RAM')
    parser.add_argument('--full_pattern', action='store_true', default=False,
                        help='Use pattern at native size (no scaling); crop height to 75% if needed')
    parser.add_argument('--calib_error_magnitudes', type=str, default='0',
                        help='Comma-separated list of calibration error magnitudes (0,1-5). 0 = none.')
    parser.add_argument('--fg_list',
        type=str,
        default='',
        help='Comma-separated freq:gran pairs, e.g. "2:02,3:03". '
             'freq matches size_<freq>, gran matches filename stem.'
    )
    parser.add_argument(
        '--distort_benign',
        action='store_true',
        default=False,
        help='If set, apply calibration error to benign images (per calib magnitude) and save distorted-benign stats/images. Unused presently.'
    )

    args = parser.parse_args()

    device = torch.device('cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu')

    freq_list = [int(x) for x in args.frequency.split(',') if x.strip()]
    disp_list = [int(x) for x in args.discrepency.split(',') if x.strip()]
    ratio_list = [float(x) for x in args.ratios.split(',') if x.strip()]
    calib_mag_list = [int(x) for x in args.calib_error_magnitudes.split(',') if x.strip()]
    fg_filter = parse_fg_list(args.fg_list)

    print(f"[debug] Parsed freq_list={freq_list}, disp_list={disp_list}, "
          f"ratio_list={ratio_list}, calib_mag_list={calib_mag_list}, fg_filter={fg_filter}")

    if args.model == 'stackhourglass':
        model = stackhourglass(args.maxdisp)
    elif args.model == 'basic':
        model = basic(args.maxdisp)
    else:
        raise ValueError('Unknown model type')

    model = model.to(device)

    if args.loadmodel is not None:
        print(f"[debug] Loading model from {args.loadmodel}")
        state_dict = torch.load(args.loadmodel, map_location=device)
        model_state_dict = state_dict['state_dict']
        if any(k.startswith('module.') for k in model_state_dict):
            model_state_dict = {k[7:] if k.startswith('module.') else k: v
                                for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)

    normal_mean_var = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normal_mean_var)])

    os.makedirs(args.results_dir, exist_ok=True)
    raw_disparity_dir = os.path.join(args.results_dir, 'raw_disparity')
    disparity_maps_dir = os.path.join(args.results_dir, 'disparity_maps')
    benign_patterns_dir = os.path.join(args.results_dir, 'benign_patterns')
    os.makedirs(raw_disparity_dir, exist_ok=True)
    os.makedirs(disparity_maps_dir, exist_ok=True)
    os.makedirs(benign_patterns_dir, exist_ok=True)

    # Build candidate list
    if not args.white_test:
        id_name_pairs = collect_pairs_by_numeric_id(args.left_dir, args.right_dir)
        if not id_name_pairs:
            print('No matching image names found!')
            return
        print(f"[debug] Collected {len(id_name_pairs)} candidate pairs from left_dir/right_dir")

        rng = None
        if args.index_range:
            try:
                rng = parse_index_range(args.index_range)
            except Exception as e:
                print(f"[warn] invalid --index_range: {e}")
                rng = None

        if rng is not None:
            a, b = rng
            id_name_pairs = [(nid, name) for (nid, name) in id_name_pairs if a <= nid <= b]
            print(f"[info] applied index_range {a}-{b}: {len(id_name_pairs)} pairs")

        shard_id, shard_count = infer_array_env()
        print(f"[debug] SLURM shard_id={shard_id}, shard_count={shard_count}")
        if shard_count > 1:
            before = len(id_name_pairs)
            id_name_pairs = [(nid, name) for (nid, name) in id_name_pairs if (nid % shard_count) == shard_id]
            print(f"[info] modulo sharding: kept {len(id_name_pairs)} of {before}")

        if not id_name_pairs:
            print('[warn] no files selected after filtering')
            return

        if args.max_pairs is not None and args.max_pairs > 0:
            id_name_pairs = id_name_pairs[:args.max_pairs]
            print(f"[info] truncated to max_pairs={args.max_pairs}")

        common_names = [name for (_nid, name) in id_name_pairs]
    else:
        common_names = [f'white_test_img_{i:02d}' for i in range(args.max_pairs)]

    print(f"[debug] common_names count = {len(common_names)}")

    # External pattern modes
    stereo_pattern_mode = bool(args.pattern_dir_left and args.pattern_dir_right)
    if stereo_pattern_mode:
        print(f"[debug] Stereo pattern mode ON")
        print(f"[debug] pattern_dir_left = {args.pattern_dir_left}")
        print(f"[debug] pattern_dir_right = {args.pattern_dir_right}")

        patterns_left_all = load_external_patterns(args.pattern_dir_left)
        patterns_right_all = load_external_patterns(args.pattern_dir_right)

        print(f"[debug] loaded {len(patterns_left_all)} left patterns, "
              f"{len(patterns_right_all)} right patterns")

        if patterns_left_all:
            print(f"[debug] first 3 left patterns: {patterns_left_all[:3]}")
        if patterns_right_all:
            print(f"[debug] first 3 right patterns: {patterns_right_all[:3]}")

        def group_by_fg(lst):
            d = {}
            for p in lst:
                key = (p['freq'], p['granularity'])
                d.setdefault(key, []).append(p)
            return d

        left_group = group_by_fg(patterns_left_all)
        right_group = group_by_fg(patterns_right_all)

        print(f"[debug] left_group keys (freq,gran): {list(left_group.keys())[:10]}")
        print(f"[debug] right_group keys (freq,gran): {list(right_group.keys())[:10]}")

        stereo_specs = sorted(set(left_group.keys()) & set(right_group.keys()))
        print(f"[debug] stereo_specs (intersection) count = {len(stereo_specs)}")
        if stereo_specs:
            print(f"[debug] first stereo_specs: {stereo_specs[:10]}")

        def pick_first(kdict, key):
            return kdict[key][0]['path'] if kdict.get(key) else None

        pat_iter = [{'freq': fg[0], 'granularity': fg[1],
                     'pathL': pick_first(left_group, fg), 'pathR': pick_first(right_group, fg)}
                    for fg in stereo_specs]

        print(f"[debug] pat_iter length = {len(pat_iter)}")
        if pat_iter:
            print(f"[debug] first 5 pat_iter entries:")
            for p in pat_iter[:5]:
                print("        ", p)

        if fg_filter:
            allowed = set(fg_filter)
            before = len(pat_iter)
            pat_iter = [p for p in pat_iter if (p['freq'], p['granularity']) in allowed]
            print(f"[debug] fg_filter = {fg_filter}, filtered pat_iter {before} -> {len(pat_iter)}")

        using_external = True
        use_per_eye_dirs = False
    else:
        external_patterns = load_external_patterns(args.pattern_dir) if args.pattern_dir else []
        if external_patterns:
            print(f"[debug] Mono external pattern mode; loaded {len(external_patterns)} patterns")
            pat_iter = [{'freq': p['freq'], 'granularity': p['granularity'], 'path': p['path']}
                        for p in external_patterns]

            if fg_filter:
                allowed = set(fg_filter)
                before = len(pat_iter)
                pat_iter = [p for p in pat_iter if (p['freq'], p['granularity']) in allowed]
                print(f"[debug] fg_filter = {fg_filter}, filtered pat_iter {before} -> {len(pat_iter)}")

            using_external = True
        else:
            print(f"[debug] No external patterns found; falling back to synthetic stripes")
            pat_iter = [{'freq': f, 'granularity': None, 'path': None} for f in freq_list]
            using_external = False

        use_per_eye_dirs = (args.attack_left_dir and args.attack_right_dir and not using_external)
        print(f"[debug] using_external={using_external}, use_per_eye_dirs={use_per_eye_dirs}")
        print(f"[debug] pat_iter length = {len(pat_iter)}")

    CHUNK_SIZE = args.chunk_size

    # CSV streaming setup
    csv_path = os.path.join(args.results_dir, 'results.csv')
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

    need_header = file_is_empty(csv_path)
    mode = 'a' if not need_header else 'w'
    f_csv = open(csv_path, mode, newline='')
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    if need_header:
        writer.writeheader()
    f_csv.flush()

    # Histogram CSVs
    # Full patch-region histogram
    hist_full_csv_path = os.path.join(args.results_dir, 'hist_center_region_full.csv')
    # Inner patch-region histogram
    hist_inner_csv_path = os.path.join(args.results_dir, 'hist_center_region_inner.csv')

    hist_fields = ['pair', 'ratio', 'frequency', 'granularity', 'discrepency', 'calib_magnitude', 'phase']
    hist_fields.extend([f'd{d}' for d in range(193)])

    # Full-region histogram file
    need_hist_full_header = file_is_empty(hist_full_csv_path)
    mode_hist_full = 'a' if not need_hist_full_header else 'w'
    f_hist_full = open(hist_full_csv_path, mode_hist_full, newline='')
    hist_writer_full = csv.DictWriter(f_hist_full, fieldnames=hist_fields)
    if need_hist_full_header:
        hist_writer_full.writeheader()
    f_hist_full.flush()

    # Inner-region histogram file
    need_hist_inner_header = file_is_empty(hist_inner_csv_path)
    mode_hist_inner = 'a' if not need_hist_inner_header else 'w'
    f_hist_inner = open(hist_inner_csv_path, mode_hist_inner, newline='')
    hist_writer_inner = csv.DictWriter(f_hist_inner, fieldnames=hist_fields)
    if need_hist_inner_header:
        hist_writer_inner.writeheader()
    f_hist_inner.flush()

    done_patterns, done_pairs_by_pattern = load_progress(csv_path)
    for key, pairs in done_pairs_by_pattern.items():
        if len(pairs) >= len(common_names):
            done_patterns.add(key)

    print(f"[info] Resume: have partial/complete data for {len(done_pairs_by_pattern)} patterns; "
          f"{len(done_patterns)} are fully complete")

    total_units = len(ratio_list) * len(pat_iter) * len(disp_list) * len(calib_mag_list) * len(common_names)
    done_units = 0
    t_start = time.perf_counter()

    for ratio in ratio_list:
        print(f"[info] Processing ratio {ratio:.3f}")

        # BENIGN PASS - no pattern present.
        benign_stats_cache = {}
        benign_inner_stats_cache = {}
        # Per-ratio, per-image region metadata based on patch size on the original image
        # center_region_meta[fname] = (cy, cx, ph, pw, inner_ph, inner_pw)
        center_region_meta = {}
        for chunk_start in range(0, len(common_names), CHUNK_SIZE):
            chunk_names = common_names[chunk_start:chunk_start + CHUNK_SIZE]
            print(f"[debug] Benign chunk {chunk_start}-{chunk_start + len(chunk_names)} ({len(chunk_names)} images)")
            left_t_list, right_t_list, pad_meta = [], [], []
            imgs_chunk = []

            for fname in chunk_names:
                if args.white_test:
                    h, w = 370, 1224
                    l_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                    r_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                else:
                    l_path = os.path.join(args.left_dir, fname)
                    r_path = os.path.join(args.right_dir, fname)
                    l_img = np.array(Image.open(l_path).convert('RGB'))
                    r_img = np.array(Image.open(r_path).convert('RGB'))
                imgs_chunk.append((fname, l_img, r_img))

                # Define the patch size on the original image using the same rule
                # as the attack patch.
                h_img, w_img = l_img.shape[:2]
                ph = int(math.ceil(h_img * ratio))
                pw = int(math.ceil(w_img * ratio))
                cy = h_img // 2
                cx = w_img // 2
                inner_ph = max(1, int(round(ph * INNER_CENTER_RATIO)))
                inner_pw = max(1, int(round(pw * INNER_CENTER_RATIO)))
                center_region_meta[fname] = (cy, cx, ph, pw, inner_ph, inner_pw)

                l_t, top_pad, right_pad = to_tensor_normalized_padded(l_img, infer_transform, pad_mult=16)
                r_t, _, _ = to_tensor_normalized_padded(r_img, infer_transform, pad_mult=16)
                left_t_list.append(l_t)
                right_t_list.append(r_t)
                pad_meta.append((top_pad, right_pad))

            max_height = max(t.shape[1] for t in left_t_list + right_t_list)
            max_width = max(t.shape[2] for t in left_t_list + right_t_list)
            left_t_list = [pad_to_target(t, max_height, max_width) for t in left_t_list]
            right_t_list = [pad_to_target(t, max_height, max_width) for t in right_t_list]

            B = args.batch_size
            chunk_disp_cache = {}
            for i in range(0, len(chunk_names), B):
                batch_left = torch.stack(left_t_list[i:i + B], dim=0)
                batch_right = torch.stack(right_t_list[i:i + B], dim=0)
                disp_batch_np = forward_batched(model, batch_left, batch_right, device, use_amp=args.amp)
                for j, name in enumerate(chunk_names[i:i + B]):
                    top_pad, right_pad = pad_meta[i + j]
                    disp_unp = unpad_disparity(disp_batch_np[j], top_pad, right_pad)
                    chunk_disp_cache[name] = disp_unp

            for fname in chunk_names:
                disp_map = chunk_disp_cache[fname]
                cy, cx, ph, pw, inner_ph, inner_pw = center_region_meta[fname]

                stats_center = get_center_region_stats(disp_map, cy, cx, ph, pw)
                benign_stats_cache[fname] = stats_center

                stats_inner = get_center_region_stats(disp_map, cy, cx, inner_ph, inner_pw)
                benign_inner_stats_cache[fname] = stats_inner

                hist = center_region_histogram(disp_map, cy, cx, ph, pw, max_disp=192)
                hist_row = {
                    'pair': fname,
                    'ratio': ratio,
                    'frequency': -1,
                    'granularity': '',
                    'discrepency': -1,
                    'calib_magnitude': 0,
                    'phase': 'benign',
                }
                for d in range(193):
                    hist_row[f'd{d}'] = int(hist[d])
                hist_writer_full.writerow(hist_row)
                f_hist_full.flush()

                # benign inner-center histogram (written to dedicated inner CSV)
                hist_inner = center_region_histogram(disp_map, cy, cx, inner_ph, inner_pw, max_disp=192)
                hist_inner_row = {
                    'pair': fname,
                    'ratio': ratio,
                    'frequency': -1,
                    'granularity': '',
                    'discrepency': -1,
                    'calib_magnitude': 0,
                    'phase': 'benign',
                }
                for d in range(193):
                    hist_inner_row[f'd{d}'] = int(hist_inner[d])
                hist_writer_inner.writerow(hist_inner_row)
                f_hist_inner.flush()

                idx_global = common_names.index(fname)
                if args.save_raw_disparity:
                    np.save(os.path.join(
                        raw_disparity_dir,
                        f"img{idx_global:02d}_benign_ratio{ratio:.3f}_{fname}.npy"
                    ), disp_map)
                if (idx_global < 10) and args.save_disparity_maps:
                    disp_norm = ((disp_map - disp_map.min()) /
                                 (disp_map.max() - disp_map.min() + 1e-8) * 255).astype(np.uint8)
                    base_name, _ = os.path.splitext(fname)
                    out_name = f"img{idx_global:02d}_benign_ratio{ratio:.3f}_{base_name}.png"
                    Image.fromarray(disp_norm).save(os.path.join(
                        disparity_maps_dir,
                        out_name
                    ))

            del chunk_disp_cache, left_t_list, right_t_list, imgs_chunk
            print(f"[info] Processed benign chunk {chunk_start}:{chunk_start + len(chunk_names)}")

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

            # Optional distorted-benign pass per calib. Unused.
            distorted_benign_stats_cache = {}
            if args.distort_benign and use_calib:
                print(f"[calib] Computing distorted-benign disparity for calib_magnitude={calib_mag}")
                for chunk_start in range(0, len(common_names), CHUNK_SIZE):
                    chunk_names = common_names[chunk_start:chunk_start + CHUNK_SIZE]
                    left_t_list, right_t_list, pad_meta = [], [], []

                    for fname in chunk_names:
                        if args.white_test:
                            h, w = 370, 1224
                            l_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                            r_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                        else:
                            l_path = os.path.join(args.left_dir, fname)
                            r_path = os.path.join(args.right_dir, fname)
                            l_img = np.array(Image.open(l_path).convert('RGB'))
                            r_img = np.array(Image.open(r_path).convert('RGB'))

                        l_dist = apply_distortion_to_ideal_image(l_img, K_L_err, D_L_err)
                        r_dist = apply_distortion_to_ideal_image(r_img, K_R_err, D_R_err)

                        l_t, top_pad, right_pad = to_tensor_normalized_padded(l_dist, infer_transform, pad_mult=16)
                        r_t, _, _ = to_tensor_normalized_padded(r_dist, infer_transform, pad_mult=16)
                        left_t_list.append(l_t)
                        right_t_list.append(r_t)
                        pad_meta.append((top_pad, right_pad))

                    max_height = max(t.shape[1] for t in left_t_list + right_t_list)
                    max_width = max(t.shape[2] for t in left_t_list + right_t_list)
                    left_t_list = [pad_to_target(t, max_height, max_width) for t in left_t_list]
                    right_t_list = [pad_to_target(t, max_height, max_width) for t in right_t_list]

                    B = args.batch_size
                    for i in range(0, len(chunk_names), B):
                        batch_left = torch.stack(left_t_list[i:i + B], dim=0)
                        batch_right = torch.stack(right_t_list[i:i + B], dim=0)
                        disp_batch_np = forward_batched(model, batch_left, batch_right, device, use_amp=args.amp)
                        for j, name in enumerate(chunk_names[i:i + B]):
                            top_pad, right_pad = pad_meta[i + j]
                            disp_unp = unpad_disparity(disp_batch_np[j], top_pad, right_pad)

                            # Use the same center region as benign for this ratio/fname
                            cy, cx, ph, pw, inner_ph, inner_pw = center_region_meta[name]
                            stats_dist = get_center_region_stats(disp_unp, cy, cx, ph, pw)
                            distorted_benign_stats_cache[name] = stats_dist

                            idx_global = common_names.index(name)
                            if (idx_global < 10) and args.save_disparity_maps:
                                disp_norm = ((disp_unp - disp_unp.min()) /
                                             (disp_unp.max() - disp_unp.min() + 1e-8) * 255).astype(np.uint8)
                                out_name = (
                                    f"img{idx_global:02d}_benign_dist_ratio{ratio:.3f}"
                                    f"_calibM{calib_mag}_{name}"
                                )
                                Image.fromarray(disp_norm).save(os.path.join(
                                    disparity_maps_dir,
                                    out_name
                                ))
                            if args.save_raw_disparity:
                                np.save(os.path.join(
                                    raw_disparity_dir,
                                    f"img{idx_global:02d}_benign_dist_ratio{ratio:.3f}"
                                    f"_calibM{calib_mag}_{name}.npy"
                                ), disp_unp)

                print(f"[calib] Distorted-benign pass done for calib_magnitude={calib_mag}")
            else:
                distorted_benign_stats_cache = None

            for pat_item in pat_iter:
                if stereo_pattern_mode:
                    freq = int(pat_item['freq'])
                    granularity = pat_item['granularity']
                    path_left = pat_item['pathL']
                    path_right = pat_item['pathR']
                else:
                    freq = int(pat_item['freq'])
                    granularity = pat_item['granularity'] if using_external else None
                    path_left = path_right = pat_item.get('path', None)

                print(f"[debug] Entering pattern loop: calib={calib_mag}, "
                      f"freq={freq}, gran={granularity}, "
                      f"pathL={path_left}, pathR={path_right}")

                for disp in disp_list:
                    pat_key = (float(ratio), int(freq), granularity, int(disp), int(calib_mag))
                    if pat_key in done_patterns:
                        print(f"[debug] pat_key {pat_key} in done_patterns, skipping")
                        continue
                    print(f"[debug] Processing pat_key {pat_key}")
                    done_pairs = done_pairs_by_pattern.get(pat_key, set())

                    for chunk_start in range(0, len(common_names), CHUNK_SIZE):
                        chunk_names = common_names[chunk_start:chunk_start + CHUNK_SIZE]
                        atk_left_tensors, atk_right_tensors, atk_pad_meta = [], [], []
                        imgs_chunk = []

                        for fname in chunk_names:
                            if args.white_test:
                                h, w = 370, 1224
                                l_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                                r_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                            else:
                                l_path = os.path.join(args.left_dir, fname)
                                r_path = os.path.join(args.right_dir, fname)
                                l_img = np.array(Image.open(l_path).convert('RGB'))
                                r_img = np.array(Image.open(r_path).convert('RGB'))
                            imgs_chunk.append((fname, l_img, r_img))

                        for fname, l_img, r_img in imgs_chunk:
                            h_img, w_img = l_img.shape[:2]
                            cx, cy = w_img // 2, h_img // 2

                            ph = int(math.ceil(h_img * ratio))
                            pw = int(math.ceil(w_img * ratio))

                            if args.full_pattern and using_external and path_left:
                                base_img = Image.open(path_left).convert('RGB')
                                pat_full = np.array(base_img, dtype=np.uint8)
                                ph_full, pw_full = pat_full.shape[:2]
                                max_ph = int(0.75 * h_img)
                                if ph_full > max_ph:
                                    crop_top = (ph_full - max_ph) // 2
                                    crop_bottom = crop_top + max_ph
                                    pat_full = pat_full[crop_top:crop_bottom, :, :]
                                    ph_full = pat_full.shape[0]
                                pattern_left = pat_full

                                if path_right:
                                    base_img_r = Image.open(path_right).convert('RGB')
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

                                if args.luminance != 'none':
                                    pattern_left = prepare_pattern_from_image(
                                        pat_img_path=path_left,
                                        ph=pattern_left.shape[0],
                                        pw=pattern_left.shape[1],
                                        full_bg=l_img,
                                        luminance_mode=args.luminance,
                                        luminance_contrast=args.luminance_contrast,
                                        keep_aspect=False
                                    )
                                    if path_right:
                                        pattern_right = prepare_pattern_from_image(
                                            pat_img_path=path_right,
                                            ph=pattern_right.shape[0],
                                            pw=pattern_right.shape[1],
                                            full_bg=r_img,
                                            luminance_mode=args.luminance,
                                            luminance_contrast=args.luminance_contrast,
                                            keep_aspect=False
                                        )
                                    else:
                                        pattern_right = prepare_pattern_from_image(
                                            pat_img_path=path_left,
                                            ph=pattern_right.shape[0],
                                            pw=pattern_right.shape[1],
                                            full_bg=r_img,
                                            luminance_mode=args.luminance,
                                            luminance_contrast=args.luminance_contrast,
                                            keep_aspect=False
                                        )
                            else:
                                if using_external and path_left and args.keep_aspect:
                                    w0, h0 = Image.open(path_left).size
                                    pw = int(round(ph * (w0 / float(h0))))
                                elif use_per_eye_dirs and args.keep_aspect:
                                    left_pattern_path = _find_pattern_image_legacy_fg(args.attack_left_dir, freq, granularity)
                                    if left_pattern_path:
                                        w0, h0 = Image.open(left_pattern_path).size
                                        pw = int(round(ph * (w0 / float(h0))))

                                if stereo_pattern_mode:
                                    pattern_left = prepare_pattern_from_image(
                                        path_left, ph, pw, l_img,
                                        args.luminance, args.luminance_contrast, args.keep_aspect
                                    )
                                    pattern_right = prepare_pattern_from_image(
                                        path_right, ph, pw, r_img,
                                        args.luminance, args.luminance_contrast, args.keep_aspect
                                    )
                                elif using_external and path_left:
                                    pattern_left = prepare_pattern_from_image(
                                        path_left, ph, pw, l_img,
                                        args.luminance, args.luminance_contrast, args.keep_aspect
                                    )
                                    pattern_right = prepare_pattern_from_image(
                                        path_right if path_right else path_left, ph, pw, r_img,
                                        args.luminance, args.luminance_contrast, args.keep_aspect
                                    )
                                elif use_per_eye_dirs:
                                    left_pattern_path = _find_pattern_image_legacy_fg(args.attack_left_dir, freq, granularity)
                                    right_pattern_path = _find_pattern_image_legacy_fg(args.attack_right_dir, freq, granularity)
                                    if left_pattern_path and right_pattern_path:
                                        pattern_left = prepare_pattern_from_image(
                                            left_pattern_path, ph, pw, l_img,
                                            args.luminance, args.luminance_contrast, args.keep_aspect
                                        )
                                        pattern_right = prepare_pattern_from_image(
                                            right_pattern_path, ph, pw, r_img,
                                            args.luminance, args.luminance_contrast, args.keep_aspect
                                        )
                                    else:
                                        pattern_left = _fallback_generate(
                                            ph, pw, freq, l_img,
                                            args.luminance, args.luminance_contrast
                                        )
                                        pattern_right = _fallback_generate(
                                            ph, pw, freq, r_img,
                                            args.luminance, args.luminance_contrast
                                        )
                                else:
                                    pattern_left = _fallback_generate(
                                        ph, pw, freq, l_img,
                                        args.luminance, args.luminance_contrast
                                    )
                                    pattern_right = _fallback_generate(
                                        ph, pw, freq, r_img,
                                        args.luminance, args.luminance_contrast
                                    )

                            l_atk = place_pattern_on_image(l_img, pattern_left, cx, cy)
                            r_atk = place_pattern_on_image(r_img, pattern_right, cx - disp, cy)

                            if use_calib:
                                l_atk = apply_distortion_to_ideal_image(l_atk, K_L_err, D_L_err)
                                r_atk = apply_distortion_to_ideal_image(r_atk, K_R_err, D_R_err)

                            if args.save_disparity_maps:
                                idx_global = common_names.index(fname)
                                if idx_global < 10:
                                    base_name, _ = os.path.splitext(fname)
                                    gran_str = f"_g{granularity}" if granularity is not None else ""
                                    suffix = (
                                        f"_f{freq}_d{disp}_ratio{ratio:.3f}"
                                        f"_calibM{calib_mag}{gran_str}_{base_name}.png"
                                    )
                                    left_out = os.path.join(
                                        benign_patterns_dir,
                                        f"img{idx_global:02d}_L_attack{suffix}"
                                    )
                                    right_out = os.path.join(
                                        benign_patterns_dir,
                                        f"img{idx_global:02d}_R_attack{suffix}"
                                    )
                                    Image.fromarray(l_atk.astype(np.uint8)).save(left_out)
                                    Image.fromarray(r_atk.astype(np.uint8)).save(right_out)

                            l_t, top_pad, right_pad = to_tensor_normalized_padded(l_atk, infer_transform, pad_mult=16)
                            r_t, _, _ = to_tensor_normalized_padded(r_atk, infer_transform, pad_mult=16)
                            atk_left_tensors.append(l_t)
                            atk_right_tensors.append(r_t)
                            atk_pad_meta.append((top_pad, right_pad))

                        max_height_atk = max(t.shape[1] for t in atk_left_tensors + atk_right_tensors)
                        max_width_atk = max(t.shape[2] for t in atk_left_tensors + atk_right_tensors)
                        atk_left_tensors = [pad_to_target(t, max_height_atk, max_width_atk) for t in atk_left_tensors]
                        atk_right_tensors = [pad_to_target(t, max_height_atk, max_width_atk) for t in atk_right_tensors]

                        for i in range(0, len(chunk_names), args.batch_size):
                            batch_left = torch.stack(atk_left_tensors[i:i + args.batch_size], dim=0)
                            batch_right = torch.stack(atk_right_tensors[i:i + args.batch_size], dim=0)
                            disp_batch_np = forward_batched(model, batch_left, batch_right, device, use_amp=args.amp)

                            for j, fname in enumerate(chunk_names[i:i + args.batch_size]):
                                if fname in done_pairs:
                                    continue

                                top_pad, right_pad = atk_pad_meta[i + j]
                                attack_disp_map = unpad_disparity(disp_batch_np[j], top_pad, right_pad)

                                mean_ben, max_ben, min_ben, var_ben, std_ben, med_ben = benign_stats_cache[fname]
                                mean_ben_in, max_ben_in, min_ben_in, var_ben_in, std_ben_in, med_ben_in = \
                                    benign_inner_stats_cache[fname]

                                cy, cx, ph, pw, inner_ph, inner_pw = center_region_meta[fname]
                                mean_atk, max_atk, min_atk, var_atk, std_atk, med_atk = get_center_region_stats(
                                    attack_disp_map, cy, cx, ph, pw
                                )

                                mean_atk_in, max_atk_in, min_atk_in, var_atk_in, std_atk_in, med_atk_in = \
                                    get_center_region_stats(attack_disp_map, cy, cx, inner_ph, inner_pw)

                                # D1 scores: using fixed discrepancy (disp) as ground truth
                                # Full patch region
                                d1_full = compute_d1_score(attack_disp_map, cy, cx, ph, pw, ground_truth_disp=disp, threshold=3.0)
                                # Inner patch region
                                d1_inner = compute_d1_score(attack_disp_map, cy, cx, inner_ph, inner_pw, ground_truth_disp=disp, threshold=3.0)

                                # attack histogram over full patch region
                                hist = center_region_histogram(attack_disp_map, cy, cx, ph, pw, max_disp=192)
                                hist_row = {
                                    'pair': fname,
                                    'ratio': ratio,
                                    'frequency': freq,
                                    'granularity': granularity,
                                    'discrepency': disp,
                                    'calib_magnitude': calib_mag,
                                    'phase': 'attack',
                                }
                                for d in range(193):
                                    hist_row[f'd{d}'] = int(hist[d])
                                hist_writer_full.writerow(hist_row)
                                f_hist_full.flush()

                                # attack inner-center histogram (written to dedicated inner CSV)
                                hist_inner = center_region_histogram(
                                    attack_disp_map, cy, cx, inner_ph, inner_pw, max_disp=192
                                )
                                hist_inner_row = {
                                    'pair': fname,
                                    'ratio': ratio,
                                    'frequency': freq,
                                    'granularity': granularity,
                                    'discrepency': disp,
                                    'calib_magnitude': calib_mag,
                                    'phase': 'attack',
                                }
                                for d in range(193):
                                    hist_inner_row[f'd{d}'] = int(hist_inner[d])
                                hist_writer_inner.writerow(hist_inner_row)
                                f_hist_inner.flush()

                                idx_global = common_names.index(fname)
                                if args.save_raw_disparity:
                                    np.save(os.path.join(
                                        raw_disparity_dir,
                                        f"img{idx_global:02d}_attack_f{freq}_d{disp}_ratio{ratio:.3f}"
                                        f"_calibM{calib_mag}_{fname}.npy"
                                    ), attack_disp_map)

                                if (idx_global < 10) and args.save_disparity_maps:
                                    disp_norm = ((attack_disp_map - attack_disp_map.min()) /
                                                 (attack_disp_map.max() - attack_disp_map.min() + 1e-8)
                                                 * 255).astype(np.uint8)
                                    base_name, _ = os.path.splitext(fname)
                                    gran_str = f"_g{granularity}" if granularity is not None else ""
                                    out_name = (
                                        f"img{idx_global:02d}_attack_f{freq}_d{disp}_ratio{ratio:.3f}"
                                        f"_calibM{calib_mag}{gran_str}_{base_name}.png"
                                    )
                                    Image.fromarray(disp_norm).save(os.path.join(
                                        disparity_maps_dir,
                                        out_name
                                    ))

                                row = {
                                    'pair': fname,
                                    'ratio': ratio,
                                    'frequency': freq,
                                    'granularity': granularity,
                                    'discrepency': disp,
                                    'calib_magnitude': calib_mag,
                                    'mean_benign': float(mean_ben),
                                    'mean_attack': float(mean_atk),
                                    'mean_diff': float(mean_atk - mean_ben),
                                    'max_benign': float(max_ben),
                                    'max_attack': float(max_atk),
                                    'max_diff': float(max_atk - max_ben),
                                    'min_benign': float(min_ben),
                                    'min_attack': float(min_atk),
                                    'min_diff': float(min_atk - min_ben),
                                    'var_benign': float(var_ben),
                                    'var_attack': float(var_atk),
                                    'var_diff': float(var_atk - var_ben),
                                    'std_benign': float(std_ben),
                                    'std_attack': float(std_atk),
                                    'std_diff': float(std_atk - std_ben),
                                    'median_benign': float(med_ben),
                                    'median_attack': float(med_atk),
                                    'median_diff': float(med_atk - med_ben),
                                    # Inner-center stats
                                    'mean_benign_inner': float(mean_ben_in),
                                    'mean_attack_inner': float(mean_atk_in),
                                    'mean_diff_inner': float(mean_atk_in - mean_ben_in),
                                    'max_benign_inner': float(max_ben_in),
                                    'max_attack_inner': float(max_atk_in),
                                    'max_diff_inner': float(max_atk_in - max_ben_in),
                                    'min_benign_inner': float(min_ben_in),
                                    'min_attack_inner': float(min_atk_in),
                                    'min_diff_inner': float(min_atk_in - min_ben_in),
                                    'var_benign_inner': float(var_ben_in),
                                    'var_attack_inner': float(var_atk_in),
                                    'var_diff_inner': float(var_atk_in - var_ben_in),
                                    'std_benign_inner': float(std_ben_in),
                                    'std_attack_inner': float(std_atk_in),
                                    'std_diff_inner': float(std_atk_in - std_ben_in),
                                    'median_benign_inner': float(med_ben_in),
                                    'median_attack_inner': float(med_atk_in),
                                    'median_diff_inner': float(med_atk_in - med_ben_in),
                                    # D1 scores (using fixed discrepancy as ground truth)
                                    'd1_full': float(d1_full),
                                    'd1_inner': float(d1_inner),
                                }

                                if distorted_benign_stats_cache is not None and fname in distorted_benign_stats_cache:
                                    mean_bd, max_bd, min_bd, var_bd, std_bd, med_bd = distorted_benign_stats_cache[fname]
                                    row['mean_benign_dist'] = float(mean_bd)
                                    row['max_benign_dist'] = float(max_bd)
                                    row['min_benign_dist'] = float(min_bd)
                                    row['var_benign_dist'] = float(var_bd)
                                    row['std_benign_dist'] = float(std_bd)
                                    row['median_benign_dist'] = float(med_bd)
                                else:
                                    row['mean_benign_dist'] = ''
                                    row['max_benign_dist'] = ''
                                    row['min_benign_dist'] = ''
                                    row['var_benign_dist'] = ''
                                    row['std_benign_dist'] = ''
                                    row['median_benign_dist'] = ''

                                writer.writerow(row)
                                f_csv.flush()

                                done_pairs.add(fname)
                                done_pairs_by_pattern.setdefault(pat_key, set()).add(fname)

                                done_units += 1
                                elapsed = time.perf_counter() - t_start
                                eta_sec = (elapsed * (total_units / max(1, done_units) - 1.0)) if total_units > 0 else 0.0
                                pct = 100.0 * done_units / max(1, total_units)
                                if done_units % 50 == 0 or done_units == total_units:
                                    print(f"[progress] {done_units}/{total_units} ({pct:.1f}%) | "
                                          f"elapsed {fmt_hms(elapsed)} | ETA {fmt_hms(eta_sec)}")

                        del atk_left_tensors, atk_right_tensors, imgs_chunk

                    if len(done_pairs) >= len(common_names):
                        done_patterns.add(pat_key)

    f_csv.close()
    f_hist_full.close()
    f_hist_inner.close()
    print(f"\nResults saved to {csv_path}, {hist_full_csv_path} and {hist_inner_csv_path}")


if __name__ == '__main__':
    main()