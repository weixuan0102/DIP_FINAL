import numpy as np
import trimesh
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python sample_mesh.py <mesh_path> <output_ply_path> [num_points]")
        return

    mesh_path = sys.argv[1]
    output_path = sys.argv[2]
    num_points = int(sys.argv[3]) if len(sys.argv) > 3 else 100000

    print(f"Loading mesh from: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    
    print(f"Sampling {num_points} points...")
    # 均勻採樣
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    # 生成顏色 (預設為灰色或隨機，3DGS 訓練會很快修正它)
    colors = np.ones_like(points) * 128  # 灰色 (RGB 128)
    
    # 產生 Normals (可選，這裡先不存，標準 3DGS ply 格式主要吃 xyz 和 color)
    # 建立 PLY 格式的 header
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    print(f"Saving to: {output_path}")
    with open(output_path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    
    print("Done!")

if __name__ == "__main__":
    main()