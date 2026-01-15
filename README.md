# Pattern-Based Attacks on Stereo Depth (KITTI)

## Overview
This repo contains pattern-based attack scripts for two stereo depth models:
- PSMNet
- UniMatch

Note that some of the notation used in the code does not match that used in the paper, and we used a custom environment setup for testing on a cluster of b200 GPUs.
Depending on your hardware, you may need to do some tweaking.

## 1) Download KITTI
1. Download the KITTI Stereo Object Detection dataset.
2. Record the absolute paths to your KITTI image folders.

## 2) Unzip patterns folder
1. Unzip the provided `patterns` folder.
2. Record the absolute paths to the pattern folders.

## PSMNet

### Setup
1. Download and set up the PSMNet repository (follow upstream instructions).
   - Link: https://github.com/JiaRenChang/PSMNet
3. Download the PSMNet pretrained model from their repository:
   - Link: https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view
4. Copy the attack script into the PSMNet repo:
   - Place `Pattern_attack_PSMNet.py` into the built PSMNet repository.

### Run
A sample script is provided: `psmnet_script.sh`

1. Edit `psmnet_script.sh` and update:
   - `--left_dir` and `--right_dir` with your KITTI left/right image directory paths
   - `--pattern_dir_left` and `--pattern_dir_right` with the corresponding left/right attack pattern directory paths
   - `--load_model` with the filepath to the pretrained model
2. Place `psmnet_script.sh` into the built PSMNet repository.
3. Run:
```bash
bash psmnet_script.sh
```

## UniMatch

### Setup
1. Download and set up the UniMatch repository (follow upstream instructions).
   - Link: https://github.com/autonomousvision/unimatch
3. Download the UniMatch pretrained model from their repository:
   - Link: https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth
4. Copy the attack script into the PSMNet repo:
   - Place `Pattern_attack_UniMatch.py` into the built UniMatch repository.

### Run
A sample script is provided: `unimatch_script.sh`

1. Edit `psmnet_script.sh` and update:
   - `--left_dir` and `--right_dir` with your KITTI left/right image directory paths
   - `--pattern_dir_left` and `--pattern_dir_right` with the corresponding left/right attack pattern directory paths
   - `--load_model` with the filepath to the pretrained model
2. Place `unimatch_script.sh` into the built PSMNet repository.
3. Run:
```bash
bash unimatch_script.sh
```

## Results
Each test file will output three CSVs:
- `hist_center_region_full.csv`
- `hist_center_region_inner.csv`
- `results.csv`

We utilize `hist_center_region_inner.csv` in evaluation; this is a histogram of all predicted disparities within the center region of the attack pattern. The column names match the incorrect notation used in the code. Creating the heatmaps used in the paper can be done with the provided Jupyter notebook, `visualize_max_delta.ipynb`.

