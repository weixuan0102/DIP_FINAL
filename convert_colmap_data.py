import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

def main():
    # 0. Configuration
    colmap_data_dir = Path("0").absolute() # The directory with cameras.bin etc.
    output_root = Path("datasets/0448").absolute()
    temp_transforms_path = output_root / "transforms_temp.json"
    final_transforms_path = output_root / "transforms.json"
    
    colmap2k_script = Path("VideoGS/preprocess/colmap2k.py").absolute()
    
    # 1. Run colmap2k.py
    # This script will convert .bin to .text inside colmap_data_dir and generate json
    print(f"Converting COLMAP data from {colmap_data_dir}...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(colmap2k_script.parent) + ":" + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, str(colmap2k_script),
        "--text", str(colmap_data_dir),
        "--out", str(temp_transforms_path),
        "--keep_colmap_coords"
    ]
    
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error converting colmap data: {e}")
        return

    if not temp_transforms_path.exists():
        print(f"Error: {temp_transforms_path} not created.")
        return

    # 2. Filter out camera 007001
    print("Filtering out camera 007001...")
    with open(temp_transforms_path, 'r') as f:
        data = json.load(f)
    
    original_count = len(data["frames"])
    filtered_frames = []
    
    for frame in data["frames"]:
        # Check file_path for '007001'
        fpath = frame["file_path"]
        
        # 1. Filter
        if "007001" in fpath:
            print(f"Removing frame: {fpath}")
            continue
            
        # 2. Fix Filename Mismatch
        # JSON has "images/001001.png" (padded)
        # Disk has "1001.png" (unpadded/integer)
        
        path_obj = Path(fpath)
        # "images"
        parent = path_obj.parent 
        # "001001" -> 1001 -> "1001"
        stem_int = int(path_obj.stem) 
        new_name = f"{stem_int}.png"
        new_path = f"{parent}/{new_name}"
        
        frame["file_path"] = new_path
        
        filtered_frames.append(frame)
        
    data["frames"] = filtered_frames
    filtered_count = len(filtered_frames)
    
    print(f"Filtered {original_count} -> {filtered_count} frames.")
    
    # 3. Save final transforms.json
    with open(final_transforms_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    # Remove temp
    if temp_transforms_path.exists():
        temp_transforms_path.unlink()
        
    # 4. Distribute
    print("Distributing to frame folders...")
    frame_dirs = [d for d in output_root.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for fd in frame_dirs:
        dest = fd / "transforms.json"
        shutil.copy(final_transforms_path, dest)
        
    print("Done!")

if __name__ == "__main__":
    main()
