import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import multiprocessing
from functools import partial

# Global variable to store backgrounds in worker processes (via fork)
BGS = {}

def load_backgrounds(bg_dir):
    """Load all background images into the global BGS dictionary."""
    global BGS
    print("Loading backgrounds into memory (this happens once)...", flush=True)
    count = 0
    for bg_path in bg_dir.glob("*.png"):
        print(f"  Loading {bg_path.name}...", end="\r", flush=True)
        img = cv2.imread(str(bg_path))
        if img is not None:
            BGS[bg_path.name] = img.astype(np.int16)
            count += 1
    print(f"\nLoaded {count} backgrounds.")

def process_frame(frame_dir, output_dataset):
    """Process a single frame directory."""
    global BGS
    
    try:
        frame_name = frame_dir.name
        target_frame_dir = output_dataset / frame_name
        target_img_dir = target_frame_dir / "images"
        
        if target_img_dir.exists():
            shutil.rmtree(target_img_dir)
        target_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy transforms.json
        if (frame_dir / "transforms.json").exists():
            shutil.copy(frame_dir / "transforms.json", target_frame_dir / "transforms.json")
        
        src_img_dir = frame_dir / "images"
        img_paths = list(src_img_dir.glob("*.png"))
        
        # print(f"Processing Frame {frame_name} ({len(img_paths)} images)...")

        for img_path in img_paths:
            fg_name = img_path.name
            
            if fg_name not in BGS:
                # No background, copy raw
                shutil.copy(img_path, target_img_dir / fg_name)
                continue
                
            bg = BGS[fg_name] # Int16
            
            fg = cv2.imread(str(img_path))
            if fg is None:
                continue
                
            if fg.shape != bg.shape:
                bg_resized = cv2.resize(bg.astype(np.uint8), (fg.shape[1], fg.shape[0])).astype(np.int16)
                bg_use = bg_resized
            else:
                bg_use = bg
            
            fg_int = fg.astype(np.int16)
            
            # Diff and Threshold
            diff = np.abs(fg_int - bg_use)
            diff_max = np.max(diff, axis=2)
            threshold = 30
            mask = (diff_max > threshold).astype(np.uint8) * 255
            
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Composite
            alpha = mask.astype(float) / 255.0
            alpha = alpha[:, :, np.newaxis]
            white = np.full_like(fg, 255)
            out = fg.astype(float) * alpha + white.astype(float) * (1.0 - alpha)
            out = np.clip(out, 0, 255).astype(np.uint8)
            
            cv2.imwrite(str(target_img_dir / fg_name), out)
            
        print(f"Frame {frame_name} completed.")
        return frame_name
        
    except Exception as e:
        print(f"Error processing frame {frame_dir}: {e}")
        return None

def main():
    root = Path(".").absolute()
    raw_dataset = root / "datasets" / "0448_raw"
    bg_dir = root / "bg_undistorted"
    output_dataset = root / "datasets" / "0448_matting"
    
    if not bg_dir.exists():
        print("Error: bg_undistorted not found.")
        return
        
    if not output_dataset.exists():
        output_dataset.mkdir(parents=True)
        
    frames = [d for d in raw_dataset.iterdir() if d.is_dir() and d.name.isdigit()]
    frames.sort(key=lambda x: int(x.name))
    
    print(f"Found {len(frames)} frames. Init multiprocessing...")
    
    # Load BGS in main process first
    load_backgrounds(bg_dir)
    
    # Create worker arguments
    # Use 16 processes since IO is slow? Or 8? 
    # If IO is the bottleneck, more processes usually helps hide latency (if parallel access supported).
    num_workers = 12 
    
    worker_func = partial(process_frame, output_dataset=output_dataset)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(worker_func, frames)
        
    print("All matting tasks finished.")

if __name__ == "__main__":
    # Ensure 'spawn' is NOT used if we want to inherit BGS efficiently,
    # but 'fork' is default on Linux.
    main()
