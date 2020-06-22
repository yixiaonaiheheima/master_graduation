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
from models.loss import PointnetCriterion, Criterion_cross
from models.pointnet2 import PointNet2Seg
from models.pointSemantic import PointSemantic
from models.pointSemantic_dense import PointSemantic_dense
from models.pointsemantic_cross import PointSemantic_folding, PointSemantic_atlas, PointSemantic_caps
from models.PBC_cross import PBC_folding, PBC_atlas, PBC_caps
from models.PBC_pure import PBC_pure
from pointcnn_utils.pointcnn import PointCNN_seg
from models.convPoint import SegBig
from models.dgcnn import DGCNN_seg
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import json
import datetime
import multiprocessing as mp
import argparse
import time
from datetime import datetime
from utils import metric


def run_model(model, input_tensor, params, model_name, another_input=None, return_embed=False):
    """

    :param model:
    :param input_tensor: tensor(B, N, C)
    :return:
    """
    points = input_tensor[:, :, :3]
    if params['use_color'] or params['use_geometry']:
        features = input_tensor[:, :, 3:]
    else:
        features = points
    if model_name == 'pointnet':
        res = model(points.permute(0, 2, 1))
    elif model_name == 'pointnet2':
        res = model(input_tensor.permute(0, 2, 1))
    elif model_name == 'pointcnn':
        res = model(points, features)
    elif model_name == 'pointsemantic':
        return model(input_tensor, return_embed)
    elif model_name in ['pointsemantic_folding', 'pointsemantic_atlas', 'pointsemantic_caps',
                        'PBC_folding', 'PBC_atlas', 'PBC_caps']:
        if another_input is None:
            another_input = input_tensor
        res = model(input_tensor, another_input)
    elif model_name == 'convpoint':
        if not params['use_color'] and not params['use_geometry']:
            conv_features = torch.ones((points.shape[0], points.shape[1], 1)).to(points.device)
        else:
            conv_features = features
        res = model(conv_features, points)
    elif model_name == 'dgcnn':
        res = model(input_tensor.permute(0, 2, 1))
    elif model_name == 'pointsemantic_dense':
        return model(input_tensor, return_embed)
    elif model_name == 'PBC_pure':
        return model(input_tensor, return_embed)
    else:
        raise ValueError("model %s is not implemented!" % model_name)
    return res


def select_model(model_name, num_classes, params, weights=None):
    if model_name == 'pointnet':
        model = PointNetDenseCls(num_classes)
        criterion = PointnetCriterion(weights=weights)
    elif model_name == 'pointnet2':
        if params['use_geometry']:
            addition_channel = 3
        else:
            addition_channel = 0
        model = PointNet2Seg(num_classes, with_rgb=params['use_color'], addition_channel=addition_channel)
        criterion = PointnetCriterion(weights=weights)
    elif model_name == 'dgcnn':
        if params['use_geometry']:
            addition_channel = 3
        else:
            addition_channel = 0
        model = DGCNN_seg(num_classes, with_rgb=params['use_color'], addition_channel=addition_channel)
        criterion = PointnetCriterion(weights=weights)
    elif model_name == 'pointcnn':
        model = PointCNN_seg(num_classes)
        criterion = PointnetCriterion(weights=weights)
    elif model_name == 'pointsemantic':
        if params['use_geometry']:
            addition_channel = 6
        else:
            addition_channel = 0
        model = PointSemantic(num_classes, with_rgb=params['use_color'], addition_channel=addition_channel)
        criterion = PointnetCriterion(weights=weights)
    elif model_name == 'pointsemantic_dense':
        model = PointSemantic_dense(num_classes, with_rgb=params['use_color'])
        criterion = PointnetCriterion(weights=weights)
    elif model_name in ['pointsemantic_folding', 'pointsemantic_atlas', 'pointsemantic_caps',
                        'PBC_folding', 'PBC_atlas', 'PBC_caps']:
        if model_name == 'pointsemantic_folding':
            network = PointSemantic_folding
        if model_name == 'pointsemantic_atlas':
            network = PointSemantic_atlas
        if model_name == 'pointsemantic_caps':
            network = PointSemantic_caps
        if model_name == 'PBC_folding':
            network = PBC_folding
        if model_name == 'PBC_atlas':
            network = PBC_atlas
        if model_name == 'PBC_caps':
            network = PBC_caps
        if params['use_geometry']:
            addition_channel = 6
        else:
            addition_channel = 0
        model = network(num_classes, with_rgb=params['use_color'], addition_channel=addition_channel)
        criterion = Criterion_cross(weights=weights)
    elif model_name == 'convpoint':
        input_channels = 0
        if params['use_geometry']:
            input_channels += 6
        if params['use_color']:
            input_channels += 3
        if input_channels < 1:
            input_channels = 1
        model = SegBig(input_channels, num_classes, dimension=3, args={'drop': 0.5})
        criterion = PointnetCriterion(weights=weights)
    elif model_name == 'PBC_pure':
        if params['use_geometry']:
            addition_channel = 6
        else:
            addition_channel = 0
        model = PBC_pure(num_classes, with_rgb=params['use_color'], addition_channel=addition_channel)
        criterion = PointnetCriterion(weights=weights)
    else:
        raise ValueError

    return model, criterion