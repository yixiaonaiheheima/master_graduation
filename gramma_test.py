import numpy as np
import os
import h5py
import glob
import open3d

# ours: nohup python train.py -s model/ours --model PointSemantic --no_timestamp_folder > ours.out 2>&1 &
# pointnet: nohup python train_others.py --gpu_id 1 --model pointnet2 --no_timestamp_folder > pointnet2.out 2>&1 &
# pointcnn: nohup python train_others.py --gpu_id 1 --model pointcnn --batch_size_train 12 --batch_size_val 12 --no_timestamp_folder > pointcnn.out 2>&1 &
# pointnet: nohup python train_one_dataset.py --gpu_id 1 --model pointnet --batch_size_train 24 --batch_size_val 24 --no_timestamp_folder > pointnet.out 2>&1 &
# pointnet2: nohup python train_one_dataset.py --gpu_id 0 --model pointnet2 --batch_size_train 12 --s
# batch_size_val 12 --no_timestamp_folder > pointnet2.out 2>&1 &
# pointnet: nohup python train_one_dataset_with_open3d.py --gpu_id 1 --model pointnet --no_timestamp_folder --train_set train --dataset_name semantic --batch_size_train 32 --batch_size_val 32 > pointnet.out 2>&1 &
a = 5

root_folder = '/home/yss/sda1/yzl/Data/npm_raw'
file_name = 'Lille1_1.ply'
pc = open3d.io.read_point_cloud(os.path.join(root_folder, file_name))
open3d.geometry.KDTreeSearchParamHybrid

print('done')
