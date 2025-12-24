import os
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path):
    video_path = Path(video_path)
    video_name = video_path.stem
    
    # Try to parse camera ID as integer
    try:
        camera_id = int(video_name)
    except ValueError:
        camera_id = video_name
        
    cap = cv2.VideoCapture(str(video_path))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Target directory: 0448/{frame_idx}/images
        output_dir = Path(f'0448/{frame_idx}/images')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f'{camera_id}.png'
        cv2.imwrite(str(output_path), frame)
        
        frame_idx += 1
        
    cap.release()
    return f"Processed {video_name}: {frame_idx} frames"

def main():
    input_dir = '0448_video'
    video_files = sorted(glob.glob(os.path.join(input_dir, '*.mp4')))
    
    if not video_files:
        print(f"No .mp4 files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos. Starting extraction...")
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(extract_frames, video_files), total=len(video_files)))
        
    for res in results:
        print(res)
    
    print("Extraction complete.")

if __name__ == "__main__":
    main()
