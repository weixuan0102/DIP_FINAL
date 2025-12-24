import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(ret)

def main():
    # Force headless mode for Qt tasks (COLMAP)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # Configuration
    dataset_root = Path("datasets/0448")
    # Use RAW images for COLMAP (Better features from background)
    frame_0_images = Path("0448") / "0" / "images"
    
    colmap_workspace = Path("colmap_workspace_0448")
    colmap_images_dir = colmap_workspace / "images"
    colmap_db = colmap_workspace / "database.db"
    colmap_sparse = colmap_workspace / "sparse"
    colmap_text = colmap_workspace / "text"
    
    transforms_out = dataset_root / "transforms.json"
    
    # 1. Setup Workspace
    if colmap_workspace.exists():
        shutil.rmtree(colmap_workspace)
    colmap_workspace.mkdir(parents=True)
    colmap_images_dir.mkdir(parents=True)
    colmap_sparse.mkdir(parents=True)
    colmap_text.mkdir(parents=True)
    
    # Copy images from Frame 0
    print(f"Copying images from {frame_0_images}...")
    for img_file in frame_0_images.glob("*.png"):
        shutil.copy(img_file, colmap_images_dir / img_file.name)
        
    # 2. Run COLMAP
    print("Running COLMAP Feature Extraction...")
    run_command(f"colmap feature_extractor --database_path {colmap_db} --image_path {colmap_images_dir} --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 0")
    
    print("Running COLMAP Exhaustive Matcher...")
    run_command(f"colmap exhaustive_matcher --database_path {colmap_db} --SiftMatching.use_gpu 0")
    
    print("Running COLMAP Mapper...")
    # Relax Settings for sparse rig / difficult matching
    mapper_cmd = (
        f"colmap mapper --database_path {colmap_db} --image_path {colmap_images_dir} "
        f"--output_path {colmap_sparse} "
        "--Mapper.init_min_tri_angle 2 "
        "--Mapper.init_min_num_inliers 20 "
        "--Mapper.abs_pose_min_num_inliers 10 "
        "--Mapper.ba_global_images_ratio 1.3 "
    )
    run_command(mapper_cmd)
    
    # Check if sparse reconstruction exists (0 folder)
    sparse_0 = colmap_sparse / "0"
    if not sparse_0.exists():
        print("COLMAP reconstruction failed (no '0' folder in sparse output).")
        sys.exit(1)
        
    # Convert binary to text (colmap2k needs text or we can use binary reading if supported, but colmap2k.py seems to have bin2txt but expects text dir input? 
    # Actually looking at colmap2k.py, it calls bin2txt if passed folder.
    # It seems to read images.txt/cameras.txt.
    # So we should convert model to text.
    print("Converting COLMAP model to text...")
    run_command(f"colmap model_converter --input_path {sparse_0} --output_path {colmap_text} --output_type TXT")
    
    # 3. Generate transforms.json
    print("Generating transforms.json...")
    # call colmap2k.py
    # We need to make sure python finds colmap_helper.
    # colmap2k.py is in VideoGS/preprocess/
    
    colmap2k_script = Path("VideoGS/preprocess/colmap2k.py").absolute()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(colmap2k_script.parent) + ":" + env.get("PYTHONPATH", "")
    
    # Arguments based on hifi4g_process.py: --text ... --out ... --keep_colmap_coords
    # NOTE: colmap2k.py expects the directory to contain BINARY files (images.bin), 
    # and it converts them to text itself. So we point it to the sparse output (0 folder).
    cmd = [
        sys.executable, str(colmap2k_script),
        "--text", str(sparse_0),
        "--out", str(transforms_out),
        "--keep_colmap_coords"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)
    
    if not transforms_out.exists():
        print("Failed to generate transforms.json")
        sys.exit(1)
        
    # 4. Distribute to all frames
    print("Distributing transforms.json to all frames...")
    # Find all frame directories (numerical)
    frame_dirs = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for fd in frame_dirs:
        dest = fd / "transforms.json"
        shutil.copy(transforms_out, dest)
        
    print("Done!")

if __name__ == "__main__":
    main()
