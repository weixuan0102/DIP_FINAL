import os
import shutil
from pathlib import Path

def main():
    source_root = Path("0448").absolute() # Raw images
    processed_root = Path("datasets/0448").absolute() # Has transforms.json
    output_root = Path("datasets/0448_raw").absolute()
    
    transforms_src = processed_root / "transforms.json"
    
    if not transforms_src.exists():
        print(f"Error: {transforms_src} not found. Please complete previous steps first.")
        return

    if output_root.exists():
        print(f"Cleaning {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    
    # Iterate frame folders in 0448
    frame_dirs = [d for d in source_root.iterdir() if d.is_dir() and d.name.isdigit()]
    
    print(f"Processing {len(frame_dirs)} frames...")
    
    for fd in frame_dirs:
        frame_idx = fd.name
        
        # Target
        dest_dir = output_root / frame_idx / "images"
        dest_dir.mkdir(parents=True)
        
        # Copy images
        src_images = fd / "images"
        if not src_images.exists():
            print(f"Warning: {src_images} empty.")
            continue
            
        print(f"Copying frame {frame_idx}...")
        for img in src_images.glob("*.png"):
            # Only copy if it's in our filtered list?
            # User wants raw dataset. We should copy all, or only those in transforms.json?
            # Generally safe to copy all, transforms.json controls what is used.
            # But we filtered out 007001. We can skip it here too to save space.
            if "007001" in img.name:
                continue
                
            shutil.copy(img, dest_dir / img.name)
            
        # Filtered logic for 007001 checked above.
        
        # Copy transforms.json
        shutil.copy(transforms_src, output_root / frame_idx / "transforms.json")
        
    print("Done creating datasets/0448_raw")

if __name__ == "__main__":
    main()
