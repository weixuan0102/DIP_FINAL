import numpy as np
import os
import sys
import struct

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, 1)
            
            images[image_id] = (qvec, tvec, image_name)
    return images

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

def main():
    if len(sys.argv) < 3:
        print("Usage: python gen_cameras.py <sparse_dir> <base_dir>")
        return

    sparse_dir = sys.argv[1]
    base_dir = sys.argv[2]
    image_dir = os.path.join(base_dir, "images")
    
    image_bin_path = os.path.join(sparse_dir, "images.bin")
    if not os.path.exists(image_bin_path):
        print(f"Error: {image_bin_path} not found.")
        return

    print(f"Reading COLMAP from: {sparse_dir}")
    images = read_images_binary(image_bin_path)
    print(f"COLMAP contains {len(images)} images definition.")
    
    sorted_keys = sorted(images.keys(), key=lambda k: images[k][2])
    
    centers = []
    w2c_mats = []
    kept_images = 0
    
    for k in sorted_keys:
        qvec, tvec, name = images[k]
        
        # --- 自動副檔名匹配與過濾 ---
        # 1. 取得檔名 (不含副檔名)
        base_name = os.path.splitext(name)[0]
        
        # 2. 定義所有可能的檔名 (png, jpg, jpeg...)
        possible_names = [
            name,                   # 原始 (例如 001001.png)
            base_name + ".jpg",     # 嘗試 jpg
            base_name + ".jpeg",    # 嘗試 jpeg
            base_name + ".JPG",     # 嘗試 JPG
            base_name + ".PNG"      # 嘗試 PNG
        ]
        
        found = False
        for pname in possible_names:
            if os.path.exists(os.path.join(image_dir, pname)):
                found = True
                break
        
        if not found:
            # 這應該只會在 007001 發生
            print(f"⚠️ Warning: Image {name} (or .jpg variant) not found. Skipping.")
            continue
        # ---------------------------

        kept_images += 1
        R = qvec2rotmat(qvec)
        
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec
        w2c_mats.append(w2c)
        
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3])

    print(f"✅ Final processing: Kept {kept_images} valid cameras.")

    if kept_images == 0:
        print("Error: No valid images found. Check your paths and extensions.")
        return

    centers = np.array(centers)
    min_bound = centers.min(axis=0)
    max_bound = centers.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    
    radius = np.linalg.norm(centers - center, axis=1).max() * 1.1
    if radius == 0: radius = 1.0

    scale_mat = np.eye(4)
    scale_mat[:3, 3] = -center
    scale_mat[:3, :3] *= (1.0 / radius) * np.eye(3) 
    
    print(f"Calculated Center: {center}")
    print(f"Calculated Radius: {radius}")
    
    # Generate output dict
    final_dict = {}
    for i in range(len(w2c_mats)):
        c2w = np.linalg.inv(w2c_mats[i])
        final_dict[f'world_mat_{i}'] = c2w.astype(np.float32)
        final_dict[f'scale_mat_{i}'] = scale_mat.astype(np.float32)
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    npz_path = os.path.join(base_dir, "cameras_sphere.npz")
    np.savez(npz_path, **final_dict)
    print(f"Saved to {npz_path}")

if __name__ == "__main__":
    main()