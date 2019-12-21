from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
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

# Prepare inputs
num_classes = 8
print('{}-Preparing datasets...'.format(datetime.now()))
semantic3d_filelist_path = "/home/yss/sda1/yzl/Data/Semantic3D/train_data_files.txt"
npm3d_filelist_path = "/home/yss/sda1/yzl/Data/NPM3D/train_data_files.txt"
semantic3d_val_list_path = "/home/yss/sda1/yzl/Data/Semantic3D/val_data_files.txt"
npm3d_val_list_path = "/home/yss/sda1/yzl/Data/NPM3D/val_data_files.txt"

semantic3d_filelist_path = data_utils.load_seg_list(semantic3d_filelist_path)
npm3d_filelist_path = data_utils.load_seg_list(npm3d_filelist_path)

# semantic3d_data_train, _, semantic3d_data_num_train, semantic3d_label_train, _ = data_utils.load_seg(semantic3d_filelist_train)
semantic3d_data_val, semantic3d_label, semantic3d_data_num_val, semantic3d_seg_label_val, semantic3d_indices_split_to_full = data_utils.load_seg(semantic3d_val_list, 1)
# npm3d_data_val, _, npm3d_data_num_val, npm3d_label_val, _ = data_utils.load_seg(npm3d_val_list, 1)




