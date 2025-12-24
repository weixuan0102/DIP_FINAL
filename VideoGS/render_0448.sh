#!/bin/bash

# Activate environment if not already
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate videogs

echo "🎥 Starting VideoGS rendering..."

python 10_render_video.py \
    --model_path output/0448_v4_bg \
    --source_path ../datasets/0448 \
    --iteration 2000

echo "🎉 Rendering complete! Check output/0448_v4_bg/video_renders"
