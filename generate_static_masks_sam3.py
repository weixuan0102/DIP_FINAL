import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Add sam3 repo to path
sys.path.append(str(Path("sam3").absolute()))

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Error: Could not import SAM3. Make sure you are in the project root and 'sam3' folder exists.")
    sys.exit(1)

def main():
    root = Path(".").absolute()
    raw_dataset = root / "datasets" / "0448_raw"
    
    # Check if dataset exists
    if not raw_dataset.exists():
        print(f"Dataset {raw_dataset} not found.")
        return

    print("Building SAM3 Model...")
    # Assume default checkpoint download or cache
    try:
        model = build_sam3_image_model()
    except Exception as e:
        print(f"Failed to build SAM3 model: {e}")
        print("Please ensure you have access to Hugging Face SAM3 checkpoints or provide a path.")
        # Fallback to local checkpoint if user provided one?
        # Check current dir for sam3.pt?
        if (root / "sam3.pt").exists():
             print("Found sam3.pt in root, using it.")
             model = build_sam3_image_model(checkpoint_path=str(root / "sam3.pt"))
        else:
             return

    processor = Sam3Processor(model)
    
    frames = [d for d in raw_dataset.iterdir() if d.is_dir() and d.name.isdigit()]
    frames.sort(key=lambda x: int(x.name))
    
    print(f"Processing {len(frames)} frames with SAM3...")
    
    prompt_text = "person" # Assuming human subject
    
    for frame_dir in frames:
        print(f"Frame {frame_dir.name}...")
        
        mask_dir = frame_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        
        # We need to process each image in the frame folder
        # Usually one image per frame folder for VideoGS?
        # Or is it datasets/0448_raw/0/images/1001.png?
        
        src_img_dir = frame_dir / "images"
        images = list(src_img_dir.glob("*.png"))
        
        if not images:
            print(f"No images in {src_img_dir}")
            continue
            
        for img_path in images:
            img_pil = Image.open(img_path).convert("RGB")
            
            # Inference
            inference_state = processor.set_image(img_pil)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
            
            masks = output["masks"] # [N, H, W]
            scores = output["scores"]
            
            # Combine all masks for "person"
            # If multiple people, we want union.
            # Mask values are boolean or float? Usually boolean tensors.
            
            if len(masks) > 0:
                # masks: list of tensors or tensor?
                # SAM3 Image Processor returns a list of dictionaries? or a dictionary?
                # output["masks"] is a tensor [N, 1, H, W]? or [N, H, W]?
                if isinstance(masks, list):
                     print(f"Masks is list of length {len(masks)}")
                     # Maybe batch mode?
                     # Let's assume tensor [N, H, W] or [N, 1, H, W]
                     # If list of tensors...
                     pass 

                # print(f"Masks shape: {masks.shape}")
                
                # Check dims
                if masks.dim() == 4:
                    # [N, 1, H, W] -> [N, H, W]
                    masks = masks.squeeze(1)
                
                # Union of all detected "person" masks
                combined_mask = torch.any(masks, dim=0).cpu().numpy().astype(np.uint8) # [H, W]
            else:
                combined_mask = np.zeros((img_pil.height, img_pil.width), dtype=np.uint8)
            
            # Static Mask Logic:
            # Person (1) -> Dynamic -> 0 in Static Mask
            # Background (0) -> Static -> 255 in Static Mask
            
            static_mask = (1 - combined_mask) * 255
            static_mask = static_mask.astype(np.uint8)
            
            # Ensure 2D
            if static_mask.ndim > 2:
                 static_mask = static_mask.squeeze()
            
            print(f"Static mask shape: {static_mask.shape}")
            
            # Erode the static mask slightly to avoid boundary artifacts (expand the dynamic hole)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            try:
                static_mask = cv2.erode(static_mask, kernel, iterations=2)
            except Exception as e:
                print(f"Erode failed: {e}. Shape: {static_mask.shape}. Dtype: {static_mask.dtype}")
                # Fallback without erode
                pass
            
            # Save
            # Filename: same as image name? Or fixed?
            # VideoGS typically expects masks/00000.png?
            # Let's save as `static_mask.png` AND the original filename just in case.
            # User's previous script used dynamic masks maybe?
            
            cv2.imwrite(str(mask_dir / img_path.name), static_mask)
            cv2.imwrite(str(mask_dir / "static_mask.png"), static_mask)
            
    print("SAM3 Mask generation complete.")

if __name__ == "__main__":
    main()
