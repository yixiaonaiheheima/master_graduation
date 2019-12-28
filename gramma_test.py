import numpy as np
import os
import h5py
import glob
import open3d

# pointnet: nohup python train_one_dataset_with_open3d.py --gpu_id 0 --model pointnet2 --config_file semantic_no_color.json --batch_size_train 16 --batch_size_val 16 > pointnet2.out 2>&1 &
# pointsemantic: nohup python train_one_dataset_with_open3d.py --model pointsemantic --config_file semantic.json --gpu_id 1 --batch_size_train 16 --batch_size_val 16 > pointnetsemantic.out 2>&1 &
a = 5

root_folder = '/home/yss/sda1/yzl/Data/npm_raw'
file_name = 'Lille1_1.ply'
pc = open3d.io.read_point_cloud(os.path.join(root_folder, file_name))


print('done')
