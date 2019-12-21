from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../')
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
from data.augment import get_augmentations_from_list
from models.pointSemantic import PointSemantic
from models.loss import Criterion
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from datetime import datetime
# import importlib
from utils import data_utils, basics_util
import math
from metric import ConfusionMatrix
from tqdm import tqdm
from sklearn.neighbors import KDTree
import h5py


root_folder = "/home/yss/sda1/yzl/Data/Semantic3D/"
dest_folder = "/home/yss/sda1/yzl/Data/Semantic3D_add/"


def get_cov(points):
    points -= np.mean(points, axis=0)  # (n, 3)
    return points.transpose().dot(points) / points.shape[0]


def local_feat_gen(cur_list_path, is_train=False):
    with open(cur_list_path) as fid:
        h5_list = [line.strip() for line in fid.readlines()]
    for h5_file in h5_list:
        data_folder = os.path.dirname(cur_list_path)
        path = os.path.join(data_folder, h5_file)
        data, label, data_num, label_seg, indices_split_to_full = data_utils.load_h5(path)
        B, block_points, data_dim = data.shape
        # local feature is [curvature change, omni-variance, linearity, eigenvalue_entropy, normal_vertical,
        # height difference, height variance]
        local_features = np.zeros((B, block_points, 7))
        block_normals = np.zeros((B, block_points, 3))
        for batch_idx in range(B):
            N = data_num[batch_idx]
            cloud = data[batch_idx, :N, :3]  # (N, 3)
            tree = KDTree(cloud)
            neighborhoods_radius = tree.query_radius(cloud, r=0.75)
            k = min(20, N)
            _, neighborhoods_fixed = tree.query(cloud, k=k)
            neighborhoods = [neighborhoods_radius[idx] if len(neighborhoods_radius[idx]) >= len(neighborhoods_fixed[idx])
                             else neighborhoods_fixed[idx] for idx in range(N)]
            cov = np.array([get_cov(cloud[neighborhood]) for neighborhood in neighborhoods])   # (N, 3, 3)
            eigen_values, eigen_vectors = np.linalg.eigh(cov)  # (N, 3), (N, 3, 3)
            # eigen values is automatically in ascending order i.e. lambda3 <= lambda2 <= lambda1
            lambda3 = eigen_values[:, 0]  # (N,)
            lambda2 = eigen_values[:, 1]  # (N,)
            lambda1 = eigen_values[:, 2]  # (N,)
            lambda_sum = lambda1 + lambda2 + lambda3
            lambda_product = lambda1 * lambda2 * lambda3
            curvature_change = lambda3 / lambda_sum  # (N,)
            omni_variace = lambda_product ** (1/3) / lambda_sum  # (N,)
            # the following line is wrong
            # linearity = (lambda1 ** 2 - lambda2 ** 2) / lambda1 ** 2  # (N,)
            linearity = (lambda1 - lambda2) / lambda1
            eigenvalue_entropy = - (lambda1 * np.log(lambda1) + lambda2 * np.log(lambda2) + lambda3 * np.log(lambda3))
            normals = eigen_vectors[:, :, 0]  # (N, 3)
            normals_vertical_component = normals[:, -1]  # (N,)
            height_difference = np.array(
                [np.max(cloud[neighborhood, 2]) - np.min(cloud[neighborhood, 2]) for neighborhood in neighborhoods])  # (N,)
            height_variance = np.array([get_cov(cloud[neighborhood, 2]) for neighborhood in neighborhoods])  # (N,)
            local_features[batch_idx, :N, :] = np.stack(
                [curvature_change, omni_variace, linearity, eigenvalue_entropy, normals_vertical_component,
                 height_difference, height_variance], axis=1)  # （N, 7)
            block_normals[batch_idx, :N, :] = normals
        if is_train:
            dest_path = os.path.join(dest_folder, h5_file[3:])
        else:
            dest_path = os.path.join(dest_folder, h5_file)
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dest_file = h5py.File(dest_path, 'w')
        dest_file.create_dataset('data', data=data)
        dest_file.create_dataset('data_num', data=data_num)
        dest_file.create_dataset('label', data=label)
        dest_file.create_dataset('label_seg', data=label_seg)
        dest_file.create_dataset('indices_split_to_full', data=indices_split_to_full)
        dest_file.create_dataset('local_features', data=local_features)
        dest_file.create_dataset('normals', data=block_normals)
        dest_file.close()


def add_local_feat(list_fname):
    list_path = os.path.join(root_folder, list_fname)
    is_list_of_h5_list = not data_utils.is_h5_list(list_path)
    if is_list_of_h5_list:
        seg_list = data_utils.load_seg_list(list_path)
        for cur_list_path in seg_list:
            local_feat_gen(cur_list_path, is_train=True)
    else:
        cur_list_path = list_path
        local_feat_gen(cur_list_path)


list_fname = "val_data_files.txt"
add_local_feat(list_fname)
list_fname = "train_data_files.txt"
add_local_feat(list_fname)
