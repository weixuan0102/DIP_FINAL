import os
import sys
import glob
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add sam3 to sys.path
sys.path.append(os.path.abspath("sam3"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def apply_mask_and_save(image_path, mask_union, output_image_path, output_mask_path):
    # Read original image (High Res)
    image_cv = cv2.imread(str(image_path))
    if image_cv is None:
        print(f"Failed to read {image_path}")
        return

    h, w = image_cv.shape[:2]
    
    # Resize mask to original image size if needed
    if mask_union.shape != (h, w):
        mask_union = cv2.resize(mask_union.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    mask_bin = (mask_union > 0).astype(np.uint8)

    # 1. Image with white background
    white_bg = np.ones_like(image_cv) * 255
    foreground = cv2.bitwise_and(image_cv, image_cv, mask=mask_bin)
    background = cv2.bitwise_and(white_bg, white_bg, mask=(1-mask_bin))
    result_image = cv2.add(foreground, background)
    
    # Save processed image
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), result_image)

    # 2. Static Mask (Background=255, Foreground=0)
    static_mask = (1 - mask_bin) * 255
    
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask_path), static_mask)

def get_mask_from_state(output_state, img_size):
    # img_size is (W, H) of the INFERENCE image
    masks = output_state["masks"] # [N, 1, H, W] boolean tensor
    if masks.numel() > 0:
        return masks.any(dim=0).squeeze().cpu().numpy() # [H, W] bool
    else:
        return np.zeros((img_size[1], img_size[0]), dtype=bool)

def main():
    input_root = Path("0448")
    output_root = Path("datasets/0448")
    
    if not input_root.exists():
        print(f"{input_root} does not exist!")
        return

    image_files = sorted(list(input_root.glob("*/images/*.png")))
    if not image_files:
        print(f"No images found in {input_root}")
        return

    print(f"Found {len(image_files)} images to process.")

    print("Initializing SAM 3 Image Model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    # Set threshold as verified
    processor.set_confidence_threshold(0.07)
    
    pass1_prompt = "actors, person, hands, legs"
    pass2_prompts = ["wooden barrel", "sack", "bench", "gun", "prop", "object"]
    
    for img_path in tqdm(image_files, desc="Processing Images"):
        try:
            # Parse structure
            frame_dir = img_path.parent.parent
            frame_idx = frame_dir.name
            camera_filename = img_path.name 
            
            output_image_path = output_root / frame_idx / "images" / camera_filename
            output_mask_path = output_root / frame_idx / "masks" / camera_filename
            
            # Open Original
            original_image = Image.open(img_path).convert("RGB")
            
            # Resize for inference to avoid OOM (1024 max dim)
            inference_image = original_image.copy()
            inference_image.thumbnail((1024, 1024))
            inf_w, inf_h = inference_image.size
            
            # Set Image
            inference_state = processor.set_image(inference_image)
            
            # --- Pass 1: People ---
            out1 = processor.set_text_prompt(prompt=pass1_prompt, state=inference_state)
            mask1 = get_mask_from_state(out1, (inf_w, inf_h))
            
            # Clean up P1
            del out1
            torch.cuda.empty_cache()
            
            # --- Pass 2: Props (Loop) ---
            mask2_union = np.zeros_like(mask1, dtype=bool)
            
            for p in pass2_prompts:
                out_p = processor.set_text_prompt(prompt=p, state=inference_state)
                m_p = get_mask_from_state(out_p, (inf_w, inf_h))
                mask2_union = np.logical_or(mask2_union, m_p)
                
                # Clean up per prompt
                del out_p
                torch.cuda.empty_cache()
            
            # --- Combine ---
            final_mask_inference_res = np.logical_or(mask1, mask2_union)
            
            # Apply (will resize mask back to original inside)
            apply_mask_and_save(img_path, final_mask_inference_res, output_image_path, output_mask_path)
            
            # Clean up image state
            del inference_state
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print("Done!")

if __name__ == "__main__":
    main()
