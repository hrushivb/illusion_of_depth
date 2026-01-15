#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

python -u Pattern_attack_UniMatch.py \
  --left_dir "/path/to/KITTI/left" \
  --right_dir "/path/to/KITTI/right" \
  --ratios 0.25 \
  --frequency 10 \
  --load_model "/path/to/pretrained_model.pth" \
  --upsample_factor 4 \
  --num_scales 2 \
  --attn_type self_swin2d_cross_swin1d \
  --attn_splits_list 2 8 \
  --corr_radius_list -1 4 \
  --prop_radius_list -1 1 \
  --reg_refine \
  --num_reg_refine 3 \
  --max_pairs 1000000 \
  --discrepency 5,10,15,20,25,30,35 \
  --luminance mean \
  --keep_aspect \
  --save_disparity_maps \
  --save_raw_disparity \
  --results_dir "results/unimatch_attack_run" \
  --padding_factor 32 \
  --inference_size 384 1248 \
  --index_range "0-10" \ # NOTE: this is the total number of images to process.
  --pattern_dir_left "/path/to/patterns/left" \
  --pattern_dir_right "/path/to/patterns/right" \
  --shard_mode mod \
  --shard_count 8 \
  --shard_index 0 \
  > "logs/run_0.out" \
  2> "logs/run_0.err"
