import argparse
import os
import json
import numpy as np
import open3d
import time
import torch
from data.semantic_dataset import SemanticDataset
from data.npm_dataset import NpmDataset
from utils.metric import ConfusionMatrix
from utils.model_util import select_model, run_model
from utils.eval_utils import semantic2common, npm2common, _2common
from utils.point_cloud_util import _label_to_colors_by_name
import glob

dataset_name = 'semantic'
ori_folder = '/home/yss/sda1/yzl/Data/' + dataset_name + '_downsampled'
dest_folder = '/home/yss/sda1/yzl/Data/' + dataset_name + '_downsampled_colored'

os.makedirs(dest_folder, exist_ok=True)
for label_path in glob.glob(os.path.join(ori_folder, '*.labels')):
    labels = np.loadtxt(label_path, dtype=np.int64)
    label_fname = os.path.basename(label_path)
    fname_without_ext = os.path.splitext(label_fname)[0]
    pcd_fname = fname_without_ext + '.pcd'
    pcd_path = os.path.join(ori_folder, pcd_fname)
    pcd_file = open3d.io.read_point_cloud(pcd_path)
    pcd_file.colors = open3d.utility.Vector3dVector(_label_to_colors_by_name(labels, dataset_name))
    dest_fpath = os.path.join(dest_folder, pcd_fname)
    open3d.io.write_point_cloud(dest_fpath, pcd_file)
