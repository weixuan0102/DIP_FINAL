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
    datasets_root = project_root / "datasets" / "0448"
    raw_colmap_model = project_root / "0" # The provided sparse model
    
    # Workspace for undistortion
    ws = project_root / "undistort_ws"
    ws_input = ws / "input"
    ws_output = ws / "output"
    
    # Scripts
    colmap2k_script = project_root / "VideoGS/preprocess/colmap2k.py"
    
    # Maps specific to this dataset (found via inspection)
    # Disk has "1001.png", Model expects "001001.png"
    # Mapping: pad to 6 digits? '1001'.zfill(6) -> '001001'
    # Let's verify this mapping covers all 23 cameras.
    # 1001, 2001, ... 12001, ... 101001? 
    # Wait, 101001 is 6 digits. 1001 is 4.
    # Pattern: It seems the original filenames in bin are maybe just the names found in 0448?
    # User said: "001001.png" in Step 523 (convert_colmap_data output initially).
    # Then found "1001.png" on disk.
    # Model check should confirm.
    
    # 1. Setup Workspace
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    ws_input.mkdir(parents=True)
    
    # We need to construct the input directory ONCE that satisfies the model (all 23 images).
    # Since undistortion is a geometric operation based on intrinsics, 
    # and we want to apply it to EACH time frame (which has different foregrounds/masks),
    # we theoretically need to run image_undistorter for EACH frame folder.
    
    # But wait! image_undistorter outputs undistorted IMAGES.
    # It also outputs a new SPARSE MODEL (pinhole).
    # The new sparse model is CONSTANT for the rig (assuming rig doesn't deform).
    # The undistorted IMAGES change per frame.
    
    # Efficiency:
    # 1. Generate transforms.json ONCE (using a dummy run or Frame 0).
    # 2. Run image_undistorter for EACH frame to get images.
    #    Actually image_undistorter is fast. running it per frame is fine.
    
    frames = [d for d in datasets_root.iterdir() if d.is_dir() and d.name.isdigit()]
    frames.sort(key=lambda x: int(x.name))
    
    print(f"Found {len(frames)} frames to process.")
    
    # Environment for colmap2k
    env = os.environ.copy()
    env["PYTHONPATH"] = str(colmap2k_script.parent) + ":" + env.get("PYTHONPATH", "")
    env["QT_QPA_PLATFORM"] = "offscreen" # For COLMAP headless
    
    # We will compute transforms.json from the first successful undistortion and cache it
    cached_transforms_path = ws / "transforms_undistorted.json"
    transforms_generated = False
    
    for frame_dir in frames:
        print(f"Processing Frame: {frame_dir.name}")
        
        # Clear/Create Input
        if ws_input.exists():
            shutil.rmtree(ws_input)
        ws_input.mkdir()
        # Create 'images' subdir inside input, because --image_path should point to a folder CONTAINING images?
        # COLMAP usually expects --image_path to be the parent of the image files if relative paths are used? 
        # The model paths are "images/001001.png" or just "001001.png"?
        # grep result will tell us. Assuming "images/001001.png", we need ws_input/images/001001.png.
        
        input_images_dir = ws_input # Put directly in root
        # input_images_dir.mkdir() # No subdir
        
        src_images_dir = frame_dir / "images"
        
        # Link/Copy images mapping DiskName -> ModelName
        # We assume ModelName = DiskName with 0-padding to 6 digits? or just 0-padding?
        # Let's handle 007001 explicitly.
        
        # Valid cameras check (from src)
        src_files = list(src_images_dir.glob("*.png"))
        
        if not src_files:
            continue
            
        for img_path in src_files:
            name_stem = img_path.stem # e.g. 1001
            # Heuristic: Pad to 6 chars?
            # 1001 -> 001001
            # 101001 -> 101001
            model_name_stem = name_stem.zfill(6)
            model_name = f"{model_name_stem}.png"
            
            shutil.copy(img_path, input_images_dir / model_name)
            
        # Create Dummy 007001 if missing
        dummy_path = input_images_dir / "007001.png"
        if not dummy_path.exists():
            # Create black image 3840x2160 (based on previous logs)
            # Or simplified: use one likely existing image and black it out?
            # Or just create using cv2/numpy
            img = np.zeros((2160, 3840, 3), dtype=np.uint8)
            cv2.imwrite(str(dummy_path), img)
            print("Created dummy 007001.png")
            
        # Run Image Undistorter
        # Output will be in ws_output
        if ws_output.exists():
            shutil.rmtree(ws_output)
        ws_output.mkdir()
        
        cmd = [
            "colmap", "image_undistorter",
            "--image_path", str(ws_input), # This should contain the "images" folder if model has "images/name.png"
            "--input_path", str(raw_colmap_model),
            "--output_path", str(ws_output),
            "--output_type", "COLMAP",
            # Ensure we use max image size if needed, but default is fine usually
        ]
        
        # Note: --image_path behavior. 
        # If model says "images/001001.png", it looks effectively at {image_path}/images/001001.png.
        # My setup: ws_input/images/001001.png. So image_path = ws_input. Correct.
        
        run_cmd(cmd, env=env)
        
        # Result: ws_output/images/ ... ws_output/sparse/ ...
        
        # 1. Update Images in Dataset
        # Undistorted images are in ws_output/images/
        # They are named 001001.png etc.
        # We need to copy them back to frame_dir/images/ and RENAME to 1001.png (unpadded)
        
        undistorted_images = list((ws_output / "images").glob("*.png"))
        
        # Clear dest images? Or overwrite. Overwrite is safer.
        # But wait, we filtered 007001.
        
        for u_img in undistorted_images:
            name = u_img.name # 001001.png
            if "007001" in name:
                continue # Skip the dummy
            
            # Unpad
            stem = u_img.stem # 001001
            new_stem = str(int(stem)) # 1001
            new_name = f"{new_stem}.png"
            
            shutil.copy(u_img, src_images_dir / new_name)
            
        print(f"Updated images for {frame_dir.name}")
        
        # 2. Update transforms.json
        # Only needed once, but we verify per frame if we stick to copying
        if not transforms_generated:
            print("Generating transforms.json from undistorted model...")
            
            # The sparse model is in ws_output/sparse
            # But colmap2k.py needs BINARY? Or TEXT?
            # image_undistorter output_type COLMAP usually produces binary (.bin).
            # We can check or just ensure colmap2k arguments handle it.
            # My previous run of convert_colmap_data simply pointed to binary folder and it worked.
            
            sparse_out = ws_output / "sparse"
            
            cmd_2k = [
                sys.executable, str(colmap2k_script),
                "--text", str(sparse_out),
                "--out", str(cached_transforms_path),
                "--keep_colmap_coords" # Usually desirable to keep alignment
            ]
            
            run_cmd(cmd_2k, env=env)
            
            # Filter 007001 from cached json
            with open(cached_transforms_path, 'r') as f:
                data = json.load(f)
            
            filtered_frames = []
            for fr in data["frames"]:
                fp = fr["file_path"] # images/001001.png likely
                if "007001" in fp:
                    continue
                    
                # Fix filename in JSON to match Disk (unpadded)
                p_obj = Path(fp)
                new_n = f"{int(p_obj.stem)}.png"
                fr["file_path"] = f"{p_obj.parent}/{new_n}"
                
                filtered_frames.append(fr)
            data["frames"] = filtered_frames
            
            with open(cached_transforms_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            transforms_generated = True
            
        # Copy cached transforms.json to frame dir
        shutil.copy(cached_transforms_path, frame_dir / "transforms.json")
        
    print("Undistortion complete for all frames.")

if __name__ == "__main__":
    main()
