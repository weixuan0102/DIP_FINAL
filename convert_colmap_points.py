import sys
import numpy as np
from pathlib import Path
import struct
import collections

# Define minimal named tuple for point3D
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_properties[0]
            xyz = np.array(binary_point_properties[1:4])
            rgb = np.array(binary_point_properties[4:7])
            error = binary_point_properties[7]
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="II"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D

def write_ply(filename, points, colors):
    print(f"Writing {len(points)} points to {filename}...")
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            p = points[i]
            c = colors[i]
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def main():
    # Input: Undistorted Sparse Model
    sparse_dir = Path("undistort_ws_raw/output/sparse")
    bin_path = sparse_dir / "points3D.bin"
    
    if not bin_path.exists():
        print(f"Error: {bin_path} not found.")
        return
        
    # Output: Dataset Frame 0
    target_ply = Path("datasets/0448_raw/0/points3d.ply")
    
    print(f"Reading {bin_path}...")
    points3D = read_points3D_binary(str(bin_path))
    
    xyzs = []
    rgbs = []
    
    for pid, p in points3D.items():
        xyzs.append(p.xyz)
        rgbs.append(p.rgb)
        
    write_ply(str(target_ply), xyzs, rgbs)
    print("Done.")

if __name__ == "__main__":
    main()
