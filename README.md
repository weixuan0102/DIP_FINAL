# SAM3-4DGS Project

This repository contains the pipeline for **4D Gaussian Splatting (4DGS)** with **SAM3 integration** for dynamic scene reconstruction.

## 📋 Prerequisites
Ensure you have the following Conda environments installed:
- `videogs`: For running COLMAP, VideoGS training, and Rendering.
- `sam3`: For generating static masks using SAM3.
  ```bash
  git clone https://github.com/facebookresearch/sam3.git
  ```

## 🚀 1. Data Preprocessing
Before training, turn raw videos into a processed dataset.

### Step 1.0: Frame Extraction
Extract frames from source videos (`0448_video/*.mp4`) to `0448` folder.
```bash
python extract_0448.py
```

### Step 1.1: Camera Calibration (COLMAP)
Compute camera parameters.
```bash
conda activate videogs

# 1. Create intermediate folder
mkdir -p datasets/0448

# 2. Run COLMAP (Generates transforms.json)
python run_colmap_0448.py

# 3. Format conversion
python convert_colmap_data.py
```

### Step 1.2: Dataset Assembly
Assemble the final training structure.
```bash
# Creates specific folder structure in datasets/0448_raw
python create_raw_dataset.py
```

### Step 1.3: Mask Generation
Generate SAM3 masks for the assembled dataset.
```bash
conda activate sam3

# Generate masks
python generate_static_masks_sam3.py
```

---

## 🏋️ 2. Training (VideoGS)
Train the 4D Gaussian Splatting model. This process uses a **sequential training** strategy (Frame $t$ initializes from Frame $t-1$).

```bash
conda activate videogs

# Run the training script for 0448 dataset
bash VideoGS/train_0448_raw.sh
```

**Key Features:**
- **Continuity**: Initializes each frame from the previous frame's geometry.

## 🧠 Technical Details: How Training Works

The pipeline prioritizes **Temporal Consistency** and **visual stability**.

### 1. Sequential Initialization (`train_sequence.py`)
We use a **Daisy-Chain** approach:
- **Frame 0**: Initialized from COLMAP point cloud.
- **Frame $t$**: Initialized from the trained model of **Frame $t-1$**.
- This ensures the model smoothly evolves over time rather than jumping randomly.

### 2. Per-Frame Optimization (`train_dynamic.py`)
Each frame is trained for **3,000 iterations**. Key algorithms include:

#### A. Static Constraint Loss
To prevent background flickering, we use the **SAM3 Masks** (Step 1.3).
- **Dynamic Areas**: Optimized normally to match the image.
- **Static Areas (Background)**: Constrained to remain identical to Frame 0's geometry.

#### B. Adaptive Pruning (`prune_strange_points`)
Solves the "Black Screen" artifact.
- **Scale Check**: Removes Gaussians that grow too large (Scale > 2.0).
- **Drift Check**: Removes Gaussians that fly off-screen (Radius > 20.0).

#### C. Gradient Clipping
Caps gradients during backpropagation to 1.0, preventing numerical explosions.

---

## 🎭 Deep Dive: The Role of Masks

The **SAM3 Masks** act as the "Anchor" of the entire training process.

### 1. What they represent
- **Foreground (Dynamic)**: The moving person/object (Mask Value = 0, Black).
- **Background (Static)**: Everything else (Mask Value = 1, White).

### 2. How `train_dynamic.py` uses them
We implemented a custom **Static Constraint Loss** that works as follows:

1.  **Projection**: In every iteration, we project all millions of 3D Gaussians onto the current camera view.
2.  **Check**: We define a Gaussian as "Static" if it projects onto a **White Pixel** in the mask.
3.  **Constraint**:
    - If a point is **Static**, we calculate the distance between its current position $P_t$ and its position in the previous frame $P_{t-1}$.
    - We try to minimize this distance to **Zero**:
      $$ Loss_{static} = \sum_{p \in Static} || P_t - P_{t-1} ||^2 $$
    - If a point projects onto a **Black Pixel** (the person), it is ignored by this loss and is free to move.

**Result**: The background remains perfectly stable (like a tripod shot), while the person is allowed to move freely. Without this, the background would "drift" or "float" along with the person.

### Stable Video Rendering (Recommended)
Use the custom renderer that locks the camera view to avoid jitter.
```bash
conda activate videogs

# Render stable video (Fixed Camera View)
# --source_path: Path to dataset containing transforms.json
# --model_path: Path to trained output folder
python VideoGS/render_dynamic_video.py \
    --model_path VideoGS/output/0448_raw_v1 \
    --source_path datasets/0448_raw \
    --iteration 3000 \
    --fps 30
```
**Output**: `VideoGS/output/0448_raw_v1/video_renders_dynamic/dynamic_video.mp4`

---

## 🛠️ Troubleshooting

### "Black Screen" or Empty Video?
- Check `prune_strange_points` logs during training.
- Ensure `static_constraint_loss` is active.

### Strange Perspective / Jitter?
- Use (`render_dynamic_video.py`) which enforces a fixed camera.

