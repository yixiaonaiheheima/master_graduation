from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import open3d
import h5py
from utils.basics_util import compute_geometry_feature
from data.semantic_dataset import all_file_prefixes, train_file_prefixes, validation_file_prefixes


root_folder = "/home/yss/sda1/yzl/open3d_pointnet2/dataset/semantic_downsampled"
file_list = train_file_prefixes + validation_file_prefixes
for file in file_list:
    pcd_fname = file + '.pcd'
    pcd_path = os.path.join(root_folder, pcd_fname)
    if not os.path.exists(pcd_path):
        print("%s not exists!" % pcd_fname)
        continue
    h5_fname = file + '.h5'
    h5_path = os.path.join(root_folder, h5_fname)
    if os.path.exists(h5_path):
        print("%s already exists, skipping..." % h5_fname)
        continue
    pcd_file = open3d.io.read_point_cloud(pcd_path)
    cloud = np.asarray(pcd_file.points)
    geo_feature, normals = compute_geometry_feature(cloud)

    h5_file = h5py.File(h5_path, 'w')
    h5_file.create_dataset('geometry_features', data=geo_feature)
    h5_file.create_dataset('normals', data=normals)
    h5_file.close()
    print("%s finished" % h5_fname)


