from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../")
import os
import numpy as np
import open3d
import h5py
from utils.basics_util import compute_geometry_feature
from data.npm_dataset import npm_all_file_prefixes
from data.semantic_dataset import all_file_prefixes as semantic_all_file_prefixes
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='semantic', help='semantic or npm')
FLAGS = parser.parse_args()

def generate_geometry(root_folder, file_list):
    for file in file_list:
        pcd_fname = file + '.pcd'
        pcd_path = os.path.join(root_folder, pcd_fname)
        if not os.path.exists(pcd_path):
            print("%s not exists!" % pcd_fname)
            continue
        h5_fname = file + '_v3.h5'
        h5_path = os.path.join(root_folder, h5_fname)
        if os.path.exists(h5_path):
            print("%s already exists, skipping..." % h5_fname)
            continue
        pcd_file = open3d.io.read_point_cloud(pcd_path)
        cloud = np.asarray(pcd_file.points)

        print("compute geometry feature for %s" % file)
        geo_feature, normals = compute_geometry_feature(cloud)

        h5_file = h5py.File(h5_path, 'w')
        h5_file.create_dataset('geometry_features', data=geo_feature)
        h5_file.create_dataset('normals', data=normals)
        h5_file.close()
        print("%s finished" % h5_fname)


if __name__ == '__main__':
    if FLAGS.dataset_name == 'semantic':
        root_folder = "/home/yss/sda1/yzl/Data/semantic_downsampled"
        file_list = semantic_all_file_prefixes
    elif FLAGS.dataset_name == 'npm':
        root_folder = "/home/yss/sda1/yzl/Data/npm_downsampled"
        file_list = npm_all_file_prefixes
    else:
        raise ValueError("dataset name error, you should input semantic or npm!")
    generate_geometry(root_folder, file_list)


