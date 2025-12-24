import os
import cv2
import numpy as np
import shutil
from pathlib import Path

def main():
    root = Path(".").absolute()
    raw_dataset = root / "datasets" / "0448_raw"
    bg_dir = root / "bg_undistorted"
    
    # 255 = Static (Background), 0 = Dynamic (Foreground)
    
    if not bg_dir.exists():
        print("Error: bg_undistorted not found.")
        return
    
    frames = [d for d in raw_dataset.iterdir() if d.is_dir() and d.name.isdigit()]
    frames.sort(key=lambda x: int(x.name))
    
    print(f"Generating static masks for {len(frames)} frames...")
    
    # Load BGs
    bgs = {}
    for bg_path in bg_dir.glob("*.png"):
        bgs[bg_path.name] = cv2.imread(str(bg_path)).astype(np.int16)
        
    for frame_dir in frames:
        print(f"Frame {frame_dir.name}...")
        
        mask_dir = frame_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        
        src_img_dir = frame_dir / "images"
        for img_path in src_img_dir.glob("*.png"):
            fg_name = img_path.name
            
            # Default to all dynamic (0) if no BG, or all static? 
            # If no BG, we can't constrain. Assume dynamic (0).
            if fg_name not in bgs:
                print(f"Warning: No BG for {fg_name}")
                static_mask = np.zeros((1080, 1920), dtype=np.uint8) # Fallback size
                cv2.imwrite(str(mask_dir / "static_mask.png"), static_mask)
                continue
                
            fg = cv2.imread(str(img_path))
            bg = bgs[fg_name]
            
            if fg.shape != bg.shape:
                bg = cv2.resize(bg.astype(np.uint8), (fg.shape[1], fg.shape[0])).astype(np.int16)
            
            fg_int = fg.astype(np.int16)
            
            # Difference
            diff = np.abs(fg_int - bg)
            diff_max = np.max(diff, axis=2)
            
            # Threshold for Dynamic
            # > 30 = Dynamic (0)
            # <= 30 = Static (255)
            threshold = 30
            
            # 1 where static
            is_static = (diff_max <= threshold).astype(np.uint8)
            
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            # Erode static regions (expand dynamic holes) to be safe
            is_static = cv2.erode(is_static, kernel, iterations=2)
            
            static_mask = is_static * 255
            
            # Filename: simplistic assumption 1 image per frame folder? 
            # 0448_raw/0/images/1001.png
            # We save to 0448_raw/0/masks/static_mask.png (VideoGS usually expects one mask per camera?)
            # But here we are iterating frames.
            # VideoGS `scene` loader might need "1001.png" in masks folder?
            # Let's write `1001.png` to masks folder.
            
            cv2.imwrite(str(mask_dir / fg_name), static_mask)
            
    print("Mask generation complete.")

if __name__ == "__main__":
    main()
