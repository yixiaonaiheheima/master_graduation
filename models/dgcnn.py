#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, N, 1)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (B, 1, 1)

    idx = idx + idx_base  # (batch_size, num_points, k)

    idx = idx.view(-1)  # (B*N*k)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (B*N*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (B, N, k, num_dims)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)  # (B, num_dims*2, N, k)

    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        """

        :param x: tensor(B, 3, N)
        :return:
        """
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (B, 6, N, k)
        x = self.conv1(x)  # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = get_graph_feature(x1, k=self.k)  # (B, 64*2, N, k)
        x = self.conv2(x)  # (B, 64, N, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = get_graph_feature(x2, k=self.k)  # (B, 64*2, N, k)
        x = self.conv3(x)  # (B, 128, N, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        x = get_graph_feature(x3, k=self.k)  # (B, 128*2, N, k)
        x = self.conv4(x)  # (B, 256, N, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (B, 256, N)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        x = self.conv5(x)  # (B, emb_dim, N)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (B, emb_dim)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (B, emb_dim)
        x = torch.cat((x1, x2), 1)  # (B, 2*emb_dim)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (B, 512)
        x = self.dp1(x)  # (B, 512)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (B, 256)
        x = self.dp2(x)  # (B, 256)
        x = self.linear3(x)  # (B, output_channels)
        return x


class EdgeConv(nn.Module):
    def __init__(self, in_channel, channel_list):
        super(EdgeConv, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in channel_list:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, x, k):
        """

        :param x: tensor(B, num_dim, N)
        :return: tensor(B, 2*dim, N)
        """
        x = get_graph_feature(x, k=k)  # (B, 2*num_dim, N, k)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.leaky_relu(bn(conv(x)), negative_slope=0.2)  # (B, mlp[-1], N, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, mlp[-1], N)
        return x


class DGCNN_seg(nn.Module):
    def __init__(self, output_channels=40, with_rgb=False, addition_channel=0):
        super(DGCNN_seg, self).__init__()
        self.with_rgb = with_rgb
        self.feature_channel = addition_channel
        if with_rgb:
            self.feature_channel += 3
        print("feature channel for dgcnn is %d" % self.feature_channel)
        self.edge_conv1 = EdgeConv(2 * (3+self.feature_channel), [64, 64])
        self.edge_conv2 = EdgeConv(2 * 64, [64, 64])
        self.edge_conv3 = EdgeConv(2 * 64, [64])
        emb_dims = 1024
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.conv5 = nn.Sequential(nn.Conv1d(64*3, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Conv1d(emb_dims + 64*3, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.conv7 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp = nn.Dropout(p=0.5)
        self.conv8 = nn.Conv1d(256, output_channels, kernel_size=1)

    def forward(self, x):
        """

        :param x: tensor(B, C, N)
        :return:
        """
        N = x.size(2)
        k = 20
        x1 = self.edge_conv1(x, k=k)  # (B, 64, N)
        x2 = self.edge_conv2(x1, k=k)  # (B, 64, N)
        x3 = self.edge_conv3(x2, k=k)  # (B, 64, N)

        x = torch.cat((x1, x2, x3), dim=1)  # (B, 64*3, N)
        x = self.conv5(x)  # (B, emb_dim, N)
        x4 = F.adaptive_max_pool1d(x, 1)  # (B, emb_dim, 1)
        x4 = x4.repeat(1, 1, N)  # (B, emb_dim, N)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 64*3 + emb_dim, N)

        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)  # (B, 512, N)
        x = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.2)  # (B, 256, N)
        x = self.dp(x)  # (B, 256, N)
        x = self.conv8(x)  # (B, output_channels, N)
        x = F.log_softmax(x, dim=1)  # (B, output_channels, N)
        x = x.permute(0, 2, 1)  # (B, N, output_channels)

        return x