import os
import sys
import shutil
import subprocess
import glob
from pathlib import Path
import json
import numpy as np
import cv2

def run_cmd(cmd, env=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)

def main():
    # Paths
    project_root = Path(".").absolute()
    datasets_root = project_root / "datasets" / "0448_raw" # TARGET RAW
    raw_colmap_model = project_root / "0" 
    
    # Workspace for undistortion (SEPARATE)
    ws = project_root / "undistort_ws_raw"
    ws_input = ws / "input"
    ws_output = ws / "output"
    
    # Scripts
    colmap2k_script = project_root / "VideoGS/preprocess/colmap2k.py"
    
    # 1. Setup Workspace
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    ws_input.mkdir(parents=True)
    
    frames = [d for d in datasets_root.iterdir() if d.is_dir() and d.name.isdigit()]
    frames.sort(key=lambda x: int(x.name))
    
    print(f"Found {len(frames)} frames to process in 0448_raw.")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(colmap2k_script.parent) + ":" + env.get("PYTHONPATH", "")
    env["QT_QPA_PLATFORM"] = "offscreen" 
    
    cached_transforms_path = ws / "transforms_undistorted.json"
    transforms_generated = False
    
    for frame_dir in frames:
        print(f"Processing Frame: {frame_dir.name}")
        
        # Clear/Create Input
        if ws_input.exists():
            shutil.rmtree(ws_input)
        ws_input.mkdir()
        input_images_dir = ws_input # Root
        
        src_images_dir = frame_dir / "images"
        
        src_files = list(src_images_dir.glob("*.png"))
        
        if not src_files:
            continue
            
        for img_path in src_files:
            name_stem = img_path.stem 
            model_name_stem = name_stem.zfill(6)
            model_name = f"{model_name_stem}.png"
            shutil.copy(img_path, input_images_dir / model_name)
            
        dummy_path = input_images_dir / "007001.png"
        if not dummy_path.exists():
            img = np.zeros((2160, 3840, 3), dtype=np.uint8)
            cv2.imwrite(str(dummy_path), img)
            
        if ws_output.exists():
            shutil.rmtree(ws_output)
        ws_output.mkdir()
        
        cmd = [
            "colmap", "image_undistorter",
            "--image_path", str(ws_input),
            "--input_path", str(raw_colmap_model),
            "--output_path", str(ws_output),
            "--output_type", "COLMAP",
        ]
        
        run_cmd(cmd, env=env)
        
        # Update Images
        undistorted_images = list((ws_output / "images").glob("*.png"))
        
        for u_img in undistorted_images:
            name = u_img.name
            if "007001" in name:
                continue
            
            stem = u_img.stem 
            new_stem = str(int(stem)) 
            new_name = f"{new_stem}.png"
            
            shutil.copy(u_img, src_images_dir / new_name)
            
        print(f"Updated images for {frame_dir.name}")
        
        # Update transforms.json
        if not transforms_generated:
            print("Generating transforms.json from undistorted model...")
            
            sparse_out = ws_output / "sparse"
            
            cmd_2k = [
                sys.executable, str(colmap2k_script),
                "--text", str(sparse_out),
                "--out", str(cached_transforms_path),
                "--keep_colmap_coords"
            ]
            
            run_cmd(cmd_2k, env=env)
            
            with open(cached_transforms_path, 'r') as f:
                data = json.load(f)
            
            filtered_frames = []
            for fr in data["frames"]:
                fp = fr["file_path"]
                if "007001" in fp:
                    continue
                p_obj = Path(fp)
                new_n = f"{int(p_obj.stem)}.png"
                fr["file_path"] = f"{p_obj.parent}/{new_n}"
                filtered_frames.append(fr)
            data["frames"] = filtered_frames
            
            with open(cached_transforms_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            transforms_generated = True
            
        shutil.copy(cached_transforms_path, frame_dir / "transforms.json")
        
    print("Undistortion complete for 0448_raw.")

if __name__ == "__main__":
    main()
