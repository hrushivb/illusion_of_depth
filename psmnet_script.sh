#!/usr/bin/env bash
set -euo pipefail

mkdir -p Results/psmnet_run

python -u Pattern_attack_PSMNet.py \
  --left_dir "/path/to/KITTI/left" \
  --right_dir "/path/to/KITTI/right" \
  --results_dir "Results/psmnet_run" \
  --model stackhourglass \
  --loadmodel "/path/to/pretrained_model_KITTI2015.tar" \
  --max_pairs 100 \ # NOTE: this is the total number of images to process.
  --discrepency "5,10,15,20,25,30,35" \
  --ratios 0.25 \
  --batch_size 40 \ # Reduce batch size if needed to avoid out of memory errors.
  --luminance mean \
  --keep_aspect \
  --pattern_dir_left "/path/to/patterns/left" \
  --pattern_dir_right "/path/to/patterns/right"

