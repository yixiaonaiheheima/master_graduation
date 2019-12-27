from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # (b, 64, n)
        x = F.relu(self.bn2(self.conv2(x)))  # (b, 128, n)
        x = F.relu(self.bn3(self.conv3(x)))  # (b, 1024, n)
        x = torch.max(x, 2, keepdim=True)[0]  # (b, 1024, 1)
        x = x.view(-1, 1024)  # (b, 1024)

        x = F.relu(self.bn4(self.fc1(x)))  # (b, 512)
        x = F.relu(self.bn5(self.fc2(x)))  # (b, 256)
        x = self.fc3(x)  # (b, 9)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)  # (b, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, input_transform=True):
        super(PointNetfeat, self).__init__()
        self.input_transform = input_transform
        if input_transform:
            self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        """

        :param x: tensor(b, 3, n)
        :return:
        x: tensor(b, 1024)
        trans: tensor(b, 3, 3)
        trans_feat: tensor(b, 64, 64)
        """
        n_pts = x.size()[2]
        if self.input_transform:
            trans = self.stn(x)  # (b, 3, 3)
            x = x.transpose(2, 1)  # (b, n, 3)
            x = torch.bmm(x, trans)  # (b, n, 3)
            x = x.transpose(2, 1)  # (b, 3, n)
        x = F.relu(self.bn1(self.conv1(x)))  # (b, 64, n)

        if self.feature_transform:
            trans_feat = self.fstn(x)  # (b, 64, 64)
            x = x.transpose(2, 1)  # (b, n, 64)
            x = torch.bmm(x, trans_feat)  # (b, n, 64)
            x = x.transpose(2,1)  # (b, 64, n)
        pointfeat = x  # (b, 64, n)
        x = F.relu(self.bn2(self.conv2(x)))  # (b, 128, n)
        x = self.bn3(self.conv3(x))  # (b, 1024, n)
        x = torch.max(x, 2, keepdim=True)[0]  # (b, 1024, 1)
        x = x.view(-1, 1024)  # (b, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)  # (b, 1024, n)
            return torch.cat([x, pointfeat], 1)  # (b, 1088, n), (b, 3, 3), (b, 64, 64)


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, input_transform=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


# class PointNetDenseCls(nn.Module):
#     def __init__(self, k=2, feature_transform=False):
#         super(PointNetDenseCls, self).__init__()
#         self.k = k
#         self.feature_transform = feature_transform
#         self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
#         self.conv1 = torch.nn.Conv1d(1088, 512, 1)
#         self.conv2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)
#
#     def forward(self, x):
#         """
#
#         :param x: tensor(b, 3, n)
#         :return:
#         x: (b, n, num_classes)
#         trans: (b, 3, 3)
#         trans_feat: (b, 64, 64)
#         """
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.conv1(x)))  # (b, 512, n)
#         x = F.relu(self.bn2(self.conv2(x)))  # (b, 256, n)
#         x = F.relu(self.bn3(self.conv3(x)))  # (b, 128, n)
#         x = self.conv4(x)  # (b, k, n)
#         x = x.transpose(2, 1).contiguous()  # (b, n, k)
#         x = F.log_softmax(x.view(-1, self.k), dim=-1)  # (b*n, k)
#         x = x.view(batchsize, n_pts, self.k)  # (b, n, k)
#         return x, trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, with_rgb=False, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.with_rgb = with_rgb
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        """

        :param x: tensor(b, 3, n)
        :return:
        x: (b, n, num_classes)
        trans: (b, 3, 3)
        trans_feat: (b, 64, 64)
        """
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))  # (b, 512, n)
        x = F.relu(self.bn2(self.conv2(x)))  # (b, 256, n)
        x = F.relu(self.bn3(self.conv3(x)))  # (b, 128, n)
        x = self.conv4(x)  # (b, k, n)
        x = x.transpose(2, 1).contiguous()  # (b, n, k)
        x = F.log_softmax(x.view(-1, self.k), dim=-1)  # (b*n, k)
        x = x.view(batchsize, n_pts, self.k)  # (b, n, k)
        return x


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]  # (1, d, d)
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
