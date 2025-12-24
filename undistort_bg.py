import os
import shutil
import subprocess
import numpy as np
import cv2
from pathlib import Path

def run_cmd(cmd, env=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)

def main():
    project_root = Path(".").absolute()
    bg_src = project_root / "bg"
    raw_colmap_model = project_root / "0" 
    
    ws = project_root / "undistort_ws_bg"
    ws_input = ws / "input"
    ws_output = ws / "output"
    
    final_output = project_root / "bg_undistorted"
    
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    ws_input.mkdir(parents=True)
    
    if final_output.exists():
        shutil.rmtree(final_output)
    final_output.mkdir(parents=True)

    print("Copying BG images...")
    # Bg images are already padded e.g. 001001.png?
    # We copy them to ws_input.
    for img in bg_src.glob("*.png"):
        shutil.copy(img, ws_input / img.name)
        
    # Dummy 007001
    dummy_path = ws_input / "007001.png"
    if not dummy_path.exists():
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        cv2.imwrite(str(dummy_path), img)
        print("Created dummy 007001.png")
        
    # Undistort
    ws_output.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen" 
    
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", str(ws_input),
        "--input_path", str(raw_colmap_model),
        "--output_path", str(ws_output),
        "--output_type", "COLMAP",
    ]
    
    run_cmd(cmd, env=env)
    
    # Copy results to bg_undistorted
    # We rename them to unpadded? Or keep padded?
    # To match 0448_raw (which has 1001.png), we should rename to 1001.png.
    # BG input: 001001.png -> Output: 001001.png
    # Dest: 1001.png
    
    for u_img in (ws_output / "images").glob("*.png"):
        name = u_img.name
        if "007001" in name:
            continue
            
        stem = u_img.stem # 001001
        new_stem = str(int(stem)) # 1001
        new_name = f"{new_stem}.png"
        
        shutil.copy(u_img, final_output / new_name)
        
    print(f"Done. Undistorted BGs in {final_output}")

if __name__ == "__main__":
    main()
