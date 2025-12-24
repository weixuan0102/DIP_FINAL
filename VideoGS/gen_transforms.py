import os
import json
import numpy as np
import struct
import sys

# --- COLMAP 讀取函式 ---
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

# COLMAP 相機模型定義 (Model ID -> (Name, NumParams))
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12)
}

def main():
    if len(sys.argv) < 3:
        print("Usage: python gen_transforms.py <colmap_sparse_dir> <images_dir> <output_json>")
        return

    sparse_dir = sys.argv[1]
    images_dir = sys.argv[2]
    output_path = sys.argv[3]

    print(f"Reading COLMAP from: {sparse_dir}")
    
    # 1. 讀取 Cameras (內參)
    cameras = {}
    cam_bin_path = os.path.join(sparse_dir, "cameras.bin")
    if not os.path.exists(cam_bin_path):
        print(f"Error: {cam_bin_path} not found.")
        return

    with open(cam_bin_path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        print(f"Found {num_cameras} cameras info.")
        for _ in range(num_cameras):
            # 修正點：正確讀取 24 bytes (id:4, model:4, width:8, height:8)
            cam_props = read_next_bytes(fid, 24, "iiQQ") 
            cam_id = cam_props[0]
            model_id = cam_props[1]
            width = cam_props[2]
            height = cam_props[3]
            
            if model_id not in CAMERA_MODELS:
                print(f"Error: Unknown camera model_id {model_id}")
                return
            
            model_name, num_params = CAMERA_MODELS[model_id]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            
            cameras[cam_id] = {
                "w": width, "h": height,
                "fl_x": 0, "fl_y": 0, "cx": 0, "cy": 0,
                "k1": 0, "k2": 0, "p1": 0, "p2": 0 
            }
            
            # 參數映射
            if model_name == "SIMPLE_PINHOLE": # f, cx, cy
                cameras[cam_id]["fl_x"] = params[0]
                cameras[cam_id]["fl_y"] = params[0]
                cameras[cam_id]["cx"] = params[1]
                cameras[cam_id]["cy"] = params[2]
            elif model_name == "PINHOLE": # fx, fy, cx, cy
                cameras[cam_id]["fl_x"] = params[0]
                cameras[cam_id]["fl_y"] = params[1]
                cameras[cam_id]["cx"] = params[2]
                cameras[cam_id]["cy"] = params[3]
            elif model_name == "OPENCV": # fx, fy, cx, cy, k1, k2, p1, p2
                cameras[cam_id]["fl_x"] = params[0]
                cameras[cam_id]["fl_y"] = params[1]
                cameras[cam_id]["cx"] = params[2]
                cameras[cam_id]["cy"] = params[3]
                # 這裡略過畸變參數，因 Instant-NGP 基礎模式通常假設已去畸變或忽略
            else:
                # 其他模型暫時當作 PINHOLE 處理前四個參數 (通常前四個都是 fx,fy,cx,cy 或類似)
                # 這是為了防止崩潰的 fallback
                cameras[cam_id]["fl_x"] = params[0]
                cameras[cam_id]["fl_y"] = params[1]
                cameras[cam_id]["cx"] = params[2]
                cameras[cam_id]["cy"] = params[3]

    # 2. 讀取 Images (外參)
    frames = []
    img_bin_path = os.path.join(sparse_dir, "images.bin")
    if not os.path.exists(img_bin_path):
        print(f"Error: {img_bin_path} not found.")
        return

    with open(img_bin_path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        print(f"Found {num_reg_images} images definition.")
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, 1)

            # 檔名檢查與修正
            base_name = os.path.splitext(image_name)[0]
            found_name = None
            possible_exts = [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]
            
            # 先試原名
            if os.path.exists(os.path.join(images_dir, image_name)):
                found_name = image_name
            else:
                # 嘗試其他副檔名
                for ext in possible_exts:
                    if os.path.exists(os.path.join(images_dir, base_name + ext)):
                        found_name = base_name + ext
                        break
            
            if not found_name:
                # 默默跳過壞掉的鏡頭
                continue

            # 計算變換矩陣 (Camera to World)
            # COLMAP: World to Camera (R, t) -> P = [R|t]
            # NGP: Camera to World
            R = qvec2rotmat(qvec)
            t = tvec
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            
            # 計算 C2W
            c2w = np.linalg.inv(w2c)
            
            # Coordinate System Conversion for Instant-NGP
            # COLMAP uses OpenCV conventions: x-right, y-down, z-forward
            # Instant-NGP (transforms.json) usually expects: x-right, y-up, z-back (OpenGL) 
            # OR x-right, y-down, z-forward (OpenCV) depending on flags.
            # Most NeRF transforms.json assume OpenGL convention.
            # Let's flip Y and Z axes to convert OpenCV -> OpenGL
            flip_mat = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            c2w = np.matmul(c2w, flip_mat)

            cam = cameras[cam_id]
            frame = {
                "file_path": os.path.join("images", found_name), 
                "transform_matrix": c2w.tolist(),
                "fl_x": cam["fl_x"],
                "fl_y": cam["fl_y"],
                "cx": cam["cx"],
                "cy": cam["cy"],
                "w": int(cam["w"]),
                "h": int(cam["h"])
            }
            frames.append(frame)

    if not frames:
        print("Error: No valid frames found matching images directory.")
        return

    # 3. 計算 AABB Scale & Output
    # 使用找到的第一個相機參數作為全域參數 (NGP 有時需要)
    aabb_scale = 4

    out_data = {
        "camera_angle_x": 0, 
        "camera_angle_y": 0, 
        "fl_x": frames[0]["fl_x"],
        "fl_y": frames[0]["fl_y"],
        "cx": frames[0]["cx"],
        "cy": frames[0]["cy"],
        "w": frames[0]["w"],
        "h": frames[0]["h"],
        "aabb_scale": aabb_scale,
        "frames": frames
    }

    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=4)
    
    print(f"✅ Saved {len(frames)} frames to {output_path}")

if __name__ == "__main__":
    main()