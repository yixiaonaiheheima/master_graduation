from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data.semantic_dataset import SemanticDataset
from data.npm_dataset import NpmDataset
import sys
import os
import numpy as np
import torch
from data.augment import get_augmentations_from_list
from models.pointnet import PointNetDenseCls
from models.loss import PointnetCriterion
from models.pointnet2 import PointNet2Seg
from models.pointSemantic import PointSemantic
from pointcnn_utils.pointcnn import PointCNN_seg
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import json
import datetime
import multiprocessing as mp
import argparse
import time
from datetime import datetime
from utils import metric


def run_model(model, input_tensor, params, model_name):
    """

    :param model:
    :param input_tensor: tensor(B, N, C)
    :return:
    """
    points = input_tensor[:, :, :3]
    if params['use_color']:
        features = input_tensor[:, :, 3:]
    else:
        features = points
    if model_name == 'pointnet':
        res = model(points.permute(0, 2, 1))
    elif model_name == 'pointnet2':
        if params['use_color']:
            res, _ = model(input_tensor.permute(0, 2, 1))
        else:
            res, _ = model(points.permute(0, 2, 1))
    elif model_name == 'pointcnn':
        res = model(points, features)
    elif model_name == 'pointsemantic':
        if params['use_color']:
            res = model(input_tensor)
        else:
            res = model(points)
    else:
        raise ValueError
    return res


def select_model(model_name, num_classes, params):
    if model_name == 'pointnet':
        model = PointNetDenseCls(num_classes)
        criterion = PointnetCriterion()
    elif model_name == 'pointnet2':
        if params['use_color']:
            model = PointNet2Seg(num_classes, with_rgb=params['use_color'])
        else:
            model = PointNet2Seg(num_classes)
        criterion = PointnetCriterion()
    elif model_name == 'pointcnn':
        model = PointCNN_seg(num_classes)
        criterion = PointnetCriterion()
    elif model_name == 'pointsemantic':
        model = PointSemantic(num_classes, with_rgb=params['use_color'])
        criterion = PointnetCriterion()
    else:
        raise ValueError

    return model, criterion