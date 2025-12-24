#!/bin/bash

# Activate environment if not already (assuming user is in 'videogs')
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate videogs

echo "🚀 Starting VideoGS training on GPU 2..."


python train_sequence.py \
    --start 0 \
    --end 60 \
    --cuda 1 \
    --data /home/nas/whperidot0x66/sam2-4dgs-project/datasets/custom_hifi4g \
    --output /home/nas/whperidot0x66/sam2-4dgs-project/VideoGS/output/custom_hifi4g_v6_sam2_undistort \
    --sh 0 \
    --interval 1 \
    --group_size 60 \
    --resolution 2 \

echo "🎉 Training complete! Results in output/custom_hifi4g_v6_sam2_undistort"
