#!/bin/bash

# Activate environment if not already (assuming user is in 'videogs')
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate videogs

# Enter the directory where this script is located (VideoGS/)
cd "$(dirname "$0")" || exit

echo "🚀 Starting VideoGS training (RAW DATASET)..."


python train_sequence.py \
    --start 0 \
    --end 60 \
    --cuda 0 \
    --data ../datasets/0448_raw \
    --output output/0448_raw_v1 \
    --sh 0 \
    --interval 1 \
    --group_size 60 \
    --resolution 2 \

echo "🎉 Training complete! Results in output/0448_raw_v1"
