import numpy as np
import os
import h5py
import glob
import open3d
from tensorboardX import SummaryWriter
import torchvision
from sklearn.neighbors import KDTree

# pointnet: nohup python train_one_dataset.py --gpu_id 0 --model pointnet2 --config_file semantic_no_color.json --batch_size_train 16 --batch_size_val 16 > pointnet2.out 2>&1 &
# pointsemantic: nohup python train_one_dataset.py --model pointsemantic --config_file semantic.json --gpu_id 1 --batch_size_train 16 --batch_size_val 16 > pointnetsemantic.out 2>&1 &
# pointcnn: nohup python train_one_dataset.py --model pointcnn --dataset_name semantic --config_file semantic.json --gpu_id 1 --batch_size_train 8 --batch_size_val 8 > pointnetsemantic.out 2>&1 &
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
# !/usr/bin/env python3
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support


def func(a, b):
    return a + b


def main():
    a_args = [1, 2, 3]
    second_arg = 1
    with Pool() as pool:
        L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        N = pool.map(partial(func, b=second_arg), a_args)
        assert L == M == N


num_neighbors = 9
cloud = np.random.rand(18, 3)
view_point = np.array([0, 0, 0])
assert(len(cloud.shape) == 2)
cloud = cloud[:, :3]
N = cloud.shape[0]
tree = KDTree(cloud)
# neighborhoods_radius = tree.query_radius(cloud, r=0.75)
_, neighborhoods_fixed = tree.query(cloud, k=num_neighbors+1)  # (N, k+1)
neighborhoods = neighborhoods_fixed[:, 1:]  # (N, k)
neighborhoods_flat = neighborhoods.flatten(order='F')
cloud_expand = np.expand_dims(cloud, axis=1)  # (N, 1, 3)
p = cloud_expand - cloud[neighborhoods, :]  # (N, k, 3)
C = np.zeros((N, 6), dtype=np.float32)
C[:, 0] = np.sum(p[:, :, 0] * p[:, :, 0], axis=-1)  # (N,)
C[:, 1] = np.sum(p[:, :, 0] * p[:, :, 1], axis=-1)  # (N,)
C[:, 2] = np.sum(p[:, :, 0] * p[:, :, 2], axis=-1)  # (N,)
C[:, 3] = np.sum(p[:, :, 1] * p[:, :, 1], axis=-1)  # (N,)
C[:, 4] = np.sum(p[:, :, 1] * p[:, :, 2], axis=-1)  # (N,)
C[:, 5] = np.sum(p[:, :, 2] * p[:, :, 2], axis=-1)  # (N,)

C = C / num_neighbors
normals = np.zeros((N, 3), dtype=np.float32)
lambda_product = np.zeros((N,), dtype=np.float32)
curvature_change = np.zeros((N,), dtype=np.float32)
omni_variace = np.zeros((N,), dtype=np.float32)
linearity = np.zeros((N,), dtype=np.float32)
eigenvalue_entropy = np.zeros((N,), dtype=np.float32)
normals_vertical_component = np.zeros((N,), dtype=np.float32)
normals = np.zeros((N,), dtype=np.float32)
for i in range(0, N):
    Cmat = np.array([[C[i, 0], C[i, 1], C[i, 2]],
                     [C[i, 1], C[i, 3], C[i, 4]],
                     [C[i, 2], C[i, 4], C[i, 5]]])
    [v, d] = np.linalg.eigh(Cmat)  # (3,), (3, 3)
    idx = np.argsort(v)
    v_sorted = v[idx]
    d_sorted = d[:, idx]
    lambda3 = v_sorted[0]
    lambda2 = v_sorted[1]
    lambda1 = v_sorted[2]
    lambda_sum = lambda1 + lambda2 + lambda3
    lambda_product = lambda1 * lambda2 * lambda3
    curvature_change[i] = lambda3 / lambda_sum
    omni_variace[i] = lambda_product ** (1 / 3) / lambda_sum
    linearity[i] = (lambda1 - lambda2) / lambda1
    eigenvalue_entropy[i] = - (lambda1 * np.log(lambda1) + lambda2 * np.log(lambda2) + lambda3 * np.log(lambda3))
    normals[i, :] = d_sorted[:, 0]  # (3,)
    direction = np.sum(normals[i, :] * view_point)
    if direction > 0:
        normals[i, :] = -normals[i, :]
    normals_vertical_component[i] = normals[i, -1]
height_difference = np.array(
    [np.max(cloud[neighborhood, 2]) - np.min(cloud[neighborhood, 2]) for neighborhood in neighborhoods])  # (N,)
height_variance = np.array([get_cov(cloud[neighborhood, 2]) for neighborhood in neighborhoods])
geometry = np.stack([curvature_change, omni_variace, linearity, eigenvalue_entropy, normals_vertical_component,
                         height_difference, height_variance], axis=1)
normalized_geometry = (geometry - np.min(geometry, axis=0)) / (np.max(geometry, axis=0) - np.min(geometry, axis=0))
normalized_geometry = 1.0 / (1.0 + np.exp(-10*(normalized_geometry - np.mean(normalized_geometry, 0))))

if __name__ == "__main__":
    # freeze_support()
    # main()
    print('done')
