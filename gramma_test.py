import numpy as np
import os
import h5py
import glob
import open3d

# pointnet: nohup python train_one_dataset.py --gpu_id 0 --model pointnet2 --config_file semantic_no_color.json --batch_size_train 16 --batch_size_val 16 > pointnet2.out 2>&1 &
# pointsemantic: nohup python train_one_dataset.py --model pointsemantic --config_file semantic.json --gpu_id 1 --batch_size_train 16 --batch_size_val 16 > pointnetsemantic.out 2>&1 &
def func1():
    return (3, 4), (2, 3)

# folder = '/home/yss/sda1/yzl/Data/semantic_raw'
# h5_fname = 'bildstein_station1_xyz_intensity_rgb.h5'
# h5_path = os.path.join(folder, h5_fname)
# h5_file = h5py.File(h5_path, 'r')
# geometry = h5_file['geometry_features'][...]
# pcd_fname = 'bildstein_station1_xyz_intensity_rgb.pcd'
# pcd_path = os.path.join(folder, pcd_fname)
# pcd_file = open3d.io.read_point_cloud(pcd_path)
# color = np.asarray(pcd_file.colors)
#
# folder = '/home/yss/sda1/yzl/Data/Semantic3D_tangent_in/bildstein_station1'
# txt_fname = 'scan.txt'
# txt_path = os.path.join(folder, txt_fname)
# scan = np.genfromtxt(txt_path, delimiter=' ', max_rows=10)

(a, b) = func1()
print('done')
