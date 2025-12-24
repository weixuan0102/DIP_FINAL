import torch
import torchvision
from gaussian_renderer import render, GaussianModel
from utils.system_utils import mkdir_p
from argparse import ArgumentParser
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import math

class MiniCam:
    def __init__(self, c2w, width, height, fovx, fovy, znear, zfar):
        # c2w is 4x4 tensor
        # w2c = inverse(c2w)
        w2c = torch.inverse(c2w)
        R = w2c[:3, :3].transpose(0, 1) # Transpose R for GS convention? 
        # Wait, GS uses row-major or column-major?
        # Standard GS Camera keys: R (3x3), T (3)
        # T is translation vector.
        # world_view_transform is w2c (4x4, usually transposed for OpenGL).
        
        # Let's use the standard utils if possible.
        # R is w2c rotation. T is w2c translation.
        # R is w2c rotation. T is w2c translation.
        self.R = w2c[:3, :3].transpose(0, 1).cpu().numpy()
        self.T = w2c[:3, 3].cpu().numpy()
        
        # GS expects "world_view_transform" (4x4) and "full_proj_transform" (4x4)
        # We can compute these using utility functions if we are careful.
        # Or just implement the necessary properties.
        
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        self.znear = znear
        self.zfar = zfar
        
        self.trans = self.T
        self.scale = 1.0 #?

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def load_camera_from_json(json_path, frame_idx=0):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Assuming "frames" list, take the first one (since each folder 0/1/etc usually has one main view or we rendering the Training view?)
    # 0448_raw/0/images/ has ONE image? No, it has ALL input images for frame 0?
    # Wait, the user has 0448_raw/0/images/xxxx.png.
    # But usually for dynamic video synthesis, we want to render the trace of ONE camera moving?
    # OR we want to render the "Train Camera" view for each frame?
    # If it's monocular, there is only 1 camera per frame.
    # So `json_path` should have only 1 frame?
    # Let's check transforms.json content from earlier view.
    # It had "frames": [ ... list of files ... ].
    # For a monocular video, frame 0 transforms.json likely contains the pose for that frame's image.
    # Wait, `datasets/0448_raw/0/transforms.json` seemed to contain MANY frames (Frame 0 to Frame N?).
    # No, `datasets/0448_raw/0` had subdirs `images/`.
    # Let's assume we want to render the camera pose corresponding to the input image of that timestamp.
    # If the input is a video turned into frames, each frame `t` has 1 camera pose.
    # Where is that pose?
    # In `datasets/0448_raw/0/transforms.json`, we saw `file_path: "images/9001.png"`, etc.
    # It seems `0/transforms.json` describes the camera poses for ALL training views of Frame 0.
    # But for a monocular video, there is only 1 view per time step.
    # Does `0448_raw` have multiple views?
    # The user COLMAP'd it.
    # If it's a monocular video, we usually have Frame 0, Frame 1... each with 1 pose.
    # But `VideoGS` structure suggests `0448_raw/0/` is "Time 0", which might have multi-view if available.
    # If the user input is a SINGLE video, `0448_raw` structure is:
    # `0/`, `1/`, `2/` ... each corresponds to a timestep.
    # Inside `0/`, there is `images/`... does it have 1 image?
    # Earlier `ls` on `10/masks` showed MANY images (10001.png, 1001.png...).
    # This implies **Multi-View Video** or **Monocular Video processing that treats nearby frames as views?**
    # User said "19/22 cameras registered" for Frame 0.
    # So Frame 0 has ~20 images.
    # Frame 1 has ~20 images.
    # This is a **Multi-View Dataset**.
    
    # Objective: Render a video.
    # Usually we want a smooth camera path.
    # Simple option: Render the **first camera** of each frame? (Assuming camera 0 is the "main" view).
    # Or render a fixed camera?
    # Or render the interpolated path?
    
    # Given the user just wants to "Player" the result, rendering the **First Camera View** of each frame is the safest bet to verify reconstruction.
    
    frame_data = data['frames'][0] # Take the first camera
    
    w = data.get('w', frame_data.get('w', 1920))
    h = data.get('h', frame_data.get('h', 1080))
    
    # Intrinsics
    if 'fl_x' in data:
        fl_x = data['fl_x']
        fl_y = data['fl_y']
    else:
        fl_x = frame_data['fl_x']
        fl_y = frame_data['fl_y']
        
    fovx = focal2fov(fl_x, w)
    fovy = focal2fov(fl_y, h)
    
    # Extrinsics
    # transform_matrix is c2w (4x4)
    c2w = np.array(frame_data['transform_matrix'])
    
    # Coordinate system check:
    # COLMAP is Right-Down-Forward?
    # GS is Right-Down-Forward?
    # Nerfstudio is Right-Up-Back?
    # Usually transforms.json from colmap2nerf is OpenGL (Right-Up-Back).
    # GS requires conversion?
    # Standard GS loader converts world_view_transform.
    # Let's assume standard NeRF format (OpenGL).
    # GS Camera checks: 
    # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
    # getWorld2View2 computes the inverse.
    # Convert OpenGL c2w to GS?
    # Usually: c2w[:3, 1:3] *= -1 (invert Y/Z).
    c2w_torch = torch.tensor(c2w, dtype=torch.float32)
    
    # Apply standard NeRF -> GS conversion (flip Y and Z axes)
    # c2w_torch[:3, 1] *= -1
    # c2w_torch[:3, 2] *= -1
    # Wait, `loadCam` in standard utils does:
    # w2c = np.linalg.inv(c2w)
    # R = np.transpose(w2c[:3,:3])
    # T = w2c[:3, 3]
    # And keeps it there?
    # Let's trust the MiniCam class to do what simple GS `loadCam` usually does.
    # But note: standard `readCamerasFromTransforms` assumes `transform_matrix` is standard NeRF style.
    # We should perform the Y/Z flip if the dataset loader did it.
    
    # Let's try to match `scene/dataset_readers.py` logic?
    # It's safer to use the existing `Scene` class but FORCE it to load different JSONs.
    
    return MiniCam(c2w_torch, w, h, fovx, fovy, znear=0.01, zfar=100.0)

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    
    args.model_path = os.path.abspath(args.model_path)
    args.source_path = os.path.abspath(args.source_path)
    
    output_dir = os.path.join(args.model_path, "video_renders_dynamic")
    mkdir_p(output_dir)
    
    # Find frames
    checkpoint_root = os.path.join(args.model_path, "checkpoint")
    frames = sorted([int(x) for x in os.listdir(checkpoint_root) if x.isdigit()])
    print(f"Found {len(frames)} frames.")
    
    # ---------------------------------------------------------
    # LOAD FIXED CAMERA (from the first available frame)
    # ---------------------------------------------------------
    fixed_cam = None
    ref_frame_idx = frames[0]
    json_path = os.path.join(args.source_path, str(ref_frame_idx), "transforms.json")
    
    if not os.path.exists(json_path):
        print(f"Error: transforms.json not found for reference frame {ref_frame_idx}")
        return

    print(f"Loading FIXED camera from: {json_path}")
    with open(json_path, 'r') as f:
        cam_data = json.load(f)

    # Select camera by name (prefer '1001.png' to ensure consistency)
    target_name = "1001.png"
    cam_view = None
    for frm in cam_data['frames']:
        if os.path.basename(frm['file_path']) == target_name:
            cam_view = frm
            break
    
    if cam_view is None:
        print(f"Warning: {target_name} not found in reference frame, using first camera.")
        cam_view = cam_data['frames'][0]
        
    print(f"Selected Camera View: {os.path.basename(cam_view['file_path'])}")

    w = int(cam_view.get('w', 1920))
    h = int(cam_view.get('h', 1080))
    
    tf_mat = np.array(cam_view['transform_matrix'])
    # Flip Y/Z for NeRF->COLMAP/GS coordinate match?
    # Standard Nerfstudio exports usually need this flip.
    # Reference: VideoGS `dataset_readers.py` -> `readCamerasFromTransforms`
    # It usually handles this.
    # Let's apply the Flip: Y *= -1, Z *= -1.
    tf_mat[:3, 1] *= -1
    tf_mat[:3, 2] *= -1
    
    c2w = torch.tensor(tf_mat, dtype=torch.float32).cuda()
    
    if 'fl_x' in cam_view:
        fovx = focal2fov(cam_view['fl_x'], w)
        fovy = focal2fov(cam_view['fl_y'], h)
    else:
        fovx = focal2fov(cam_data.get('fl_x', 1000), w) # Fallback
        fovy = focal2fov(cam_data.get('fl_y', 1000), h)

    fixed_cam = MiniCam(c2w, w, h, fovx, fovy, 0.01, 100.0)
    # ---------------------------------------------------------

    
    for frame_idx in tqdm(frames):
        # 1. Load Gaussian
        ply_path = os.path.join(checkpoint_root, str(frame_idx), "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
        if not os.path.exists(ply_path):
             # Try searching max
             sub = os.path.join(checkpoint_root, str(frame_idx), "point_cloud")
             iters = [int(x.split('_')[1]) for x in os.listdir(sub) if x.startswith('iteration_')]
             if not iters:
                 print(f"Skipping frame {frame_idx} (no ply)")
                 continue
             ply_path = os.path.join(sub, f"iteration_{max(iters)}", "point_cloud.ply")
        
        # print(f"Loading PLY: {ply_path}")
        gaussians = GaussianModel(sh_degree=0) # degree doesn't matter for load? Need match ply?
        # Usually SH degree in ply is fixed.
        gaussians.load_ply(ply_path)
        
        # 2. Use Fixed Camera
        cam = fixed_cam
            
        # 3. Render
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        # Dummy Pipeline parameters
        class Pipeline:
            convert_SHs_python = False
            compute_cov3D_python = False
            debug = False
        
        try:
            render_out = render(cam, gaussians, Pipeline(), bg)["render"]
            torchvision.utils.save_image(render_out, os.path.join(output_dir, f"{frame_idx:05d}.png"))
        except Exception as e:
            print(f"Error rendering frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()

    # FFMPEG
    os.system(f"ffmpeg -y -framerate {args.fps} -i {output_dir}/%05d.png -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -c:v libx264 -pix_fmt yuv420p {output_dir}/dynamic_video.mp4")
    print(f"Saved to {output_dir}/dynamic_video.mp4")

if __name__ == "__main__":
    main()
