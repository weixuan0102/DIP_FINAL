import os
import argparse
import shutil
import pymeshlab
import open3d as o3d
import numpy as np

# group_size = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default='')
    parser.add_argument('--end', type=int, default='')
    parser.add_argument('--cuda', type=int, default='')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--sh', type=str, default='0')
    parser.add_argument('--interval', type=str, default='')
    parser.add_argument('--group_size', type=str, default='')
    parser.add_argument('--resolution', type=int, default=2)
    args = parser.parse_args()

    print(args.start, args.end)

    # os.system("conda activate torch")
    card_id = args.cuda
    data_root_path = args.data
    output_path = args.output
    sh = args.sh
    interval = int(args.interval)
    group_size = int(args.group_size)
    resolution_scale = int(args.resolution)

    # neus2_meshlab_filter_path = os.path.join(data_root_path, "luoxi_filter.mlx")

    neus2_output_path = os.path.join(output_path, "neus2_output")
    if not os.path.exists(neus2_output_path):
        os.makedirs(neus2_output_path)

    gaussian_output_path = os.path.join(output_path, "checkpoint")

    for i in range(args.start, args.end, group_size * interval):
        group_start = i
        group_end = min(i + group_size * interval, args.end) - 1
        print(group_start, group_end)
        
        frame_path = os.path.join(data_root_path, str(i))
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        frame_neus2_output_path = os.path.join(neus2_output_path, str(i))
        if not os.path.exists(frame_neus2_output_path):
            os.makedirs(frame_neus2_output_path)
        frame_neus2_ckpt_output_path = os.path.join(frame_neus2_output_path, "frame.msgpack")
        frame_neus2_mesh_output_path = os.path.join(frame_neus2_output_path, "points3d.obj")
        
        frame_points3d_output_path = os.path.join(frame_path, "points3d.ply")
        
        if os.path.exists(frame_points3d_output_path):
            print(f"Skipping NeuS2, using existing {frame_points3d_output_path}")
        else:
            # Fallback to global points3d.ply (from frame 0)
            global_points3d_path = os.path.join(data_root_path, "0", "points3d.ply")
            if os.path.exists(global_points3d_path):
                 print(f"NeuS2 disabled. Copied global point cloud from {global_points3d_path}")
                 shutil.copy(global_points3d_path, frame_points3d_output_path)
            else:
                 print(f"CRITICAL: Global point cloud not found at {global_points3d_path} and NeuS2 is disabled.")
    


        """ Gaussian """
        # generate output
        frame_model_path = os.path.join(gaussian_output_path, str(i))
        first_frame_iteration = 12000
        first_frame_save_iterations = first_frame_iteration
        first_gaussian_command = f"CUDA_VISIBLE_DEVICES={card_id} python train.py -s {frame_path} -m {frame_model_path} --iterations {first_frame_iteration} --save_iterations {first_frame_save_iterations} --sh_degree {sh} -r {resolution_scale} --port 600{card_id}"
        os.system(first_gaussian_command)

        # prune
        prune_iterations = 4000
        prune_gaussian_command = f"CUDA_VISIBLE_DEVICES={card_id} python prune_gaussian.py -s {frame_path} -m {frame_model_path} --sh_degree {sh} -r {resolution_scale} --iterations {prune_iterations}"
        os.system(prune_gaussian_command)

        # rest frame
        dynamic_command = f"CUDA_VISIBLE_DEVICES={card_id} python train_dynamic.py -s {data_root_path} -m {gaussian_output_path} --sh_degree {sh} -r {resolution_scale} --st {group_start} --ed {group_end} --interval {interval}"
        os.system(dynamic_command)

        print(f"Finish {group_start} to {group_end}")