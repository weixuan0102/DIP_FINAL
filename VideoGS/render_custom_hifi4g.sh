#!/bin/bash

# Activate environment if not already
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate videogs

echo "🎥 Starting VideoGS rendering for Custom HIFI4G..."

# Using relative paths from VideoGS directory
# Data: ../datasets/custom_hifi4g
# output: output/custom_hifi4g_v6_sam2_undistort

python 10_render_video.py \
    --model_path output/custom_hifi4g_v6_sam2_undistort \
    --source_path ../datasets/custom_hifi4g \
    --iteration 2000

echo "🎉 Rendering complete! Check output/custom_hifi4g_v6_sam2_undistort/video_renders"
