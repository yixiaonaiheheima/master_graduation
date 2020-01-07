# pylint: disable=no-member
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.basics_util import square_distance, index_points, sample_and_group, sample_and_group_all, \
    make_box, make_sphere, make_cylinder


class Attention(nn.Module):
    def __init__(self, in_channel):
        super(Attention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.normal_conv = nn.Conv2d(in_channel, 1, 1)

    def forward(self, points):
        attention = F.softplus(self.normal_conv(points), -1)  # [B, 1, N]
        # attention = torch.sigmoid(self.conv1(points.permute(0, 2, 1)))
        attention = F.normalize(attention, p=2, dim=-1)  # [B, N]
        points_weight = torch.mul(points, attention)  # [B, N, C] * [B, N, 1]
        points_attention = torch.add(points, points_weight)  # [B, N, C]

        return points_attention


class PositionAttention(nn.Module):
    def __init__(self, in_channel):
        super(PositionAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, points):
        """
            position-attention          
            input:
                 points: input [B, C, nsample, npoint]
            return:
                 points_attention: [B, D, nsample, npoint]
        """
        B, C, ns, np = points.size()
        proj_query = points.contiguous().view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        proj_key = points.contiguous().view(B, C, -1)  # [B, C, N]
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N]
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy  # [B, N, N]
        attention = F.softmax(energy_new, dim=-1)  # [B, N, N]
        proj_value = points.contiguous().view(B, C, -1)  # [B, C, N]
        points_attention = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, N]
        points_attention = points_attention.contiguous().view(B, C, ns, np)
        points_attention = self.gamma * points_attention + points  

        return points_attention


class ChannelAttention(nn.Module):
    def __init__(self, in_channel):
        super(ChannelAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, points):
        """
            channel-attention
            input:
                 points: input [B, C, nsample, npoint]
            return:
                 points_attention: [B, D, nsample, npoint]
        """
        # channel-attention
        B, C, ns, np = points.size()
        proj_query = points.contiguous().view(B, C, -1)  # [B, C, N]
        proj_key = points.points.contiguous().view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        energy = torch.bmm(proj_query, proj_key)  # [B, C, C]
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy  # [B, C, C]
        attention = F.softmax(energy_new, dim=-1)  # [B, C, C]
        proj_value = points.contiguous().view(B, C, -1)  # [B, C, N]
        points_attention = torch.bmm(attention, proj_value)  # [B, C, N]
        points_attention = points_attention.contiguous().view(B, C, ns, np)  
        points_attention = self.gamma * points_attention + points

        return points_attention


class GraphAttention(nn.Module):
    def __init__(self, in_channel):
        super(GraphAttention, self).__init__()
        self.PAT = PositionAttention(in_channel)
        self.CAT = ChannelAttention(in_channel)

    def forward(self, points):
        """
           self attention module
        """ 
        position_attention = self.PAT(points)
        channel_attention = self.CAT(points)
        points_attention = position_attention + channel_attention

        return points_attention


# class DualAttention(nn.Module):
#     def __init__(self, in_channel):
#         super(DualAttention, self).__init__()
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.query_conv = nn.Conv2d(in_channel, in_channel // 8, 1)
#         self.key_conv = nn.Conv2d(in_channel, in_channel // 8, 1)
#         self.value_conv = nn.Conv2d(in_channel, in_channel, 1)
#         self.normal_conv = nn.Conv2d(in_channel, 1, 1)

#     def forward(self, points):
#         """
#            self attention module
#            calculate attention score for each input point, similar to pn2 original segmentation model and self-attention
#             input:
#                  points: input [B, C, nsample, npoint]
#             return:
#                  points_attention: [B, D, nsample, npoint]
#         """
#         # position-attention
#         points_p = points.permute(0, 2, 1)  # [B, C, N]
#         proj_query_p = self.query_conv(points_p).permute(0, 2, 1)  # [B, N, C]
#         proj_key_p = self.key_conv(points_p)  # [B, C, N]
#         energy_p = torch.bmm(proj_query_p, proj_key_p)  # [B, N, N]
#         attention_p = F.softmax(energy_p, dim=-1)  # [B, N, N]
#         proj_value_p = self.value_conv(points_p)  # [B, C, N]
#         points_attention_p = torch.bmm(proj_value_p, attention_p.permute(0, 2, 1))  # [B, C, N]
#         points_attention_p = self.gamma * points_attention_p + points_p  # [B, C, N]

#         # channel-attention
#         points_c = points.permute(0, 2, 1)  # [B, C, N]
#         proj_query_c = points_c  # [B, C, N]
#         proj_key_c = points_c.permute(0, 2, 1)  # [B, N, C]
#         energy_c = torch.bmm(proj_query_c, proj_key_c)  # [B, C, C]
#         energy_new_c = torch.max(energy_c, -1, keepdim=True)[0].expand_as(energy_c) - energy_c  # [B, C, C]
#         attention_c = F.softmax(energy_new_c, dim=-1)  # [B, C, C]
#         proj_value_c = points_c  # [B, C, N]
#         points_attention_c = torch.bmm(attention_c, proj_value_c)  # [B, C, N]
#         points_attention_c = self.gamma * points_attention_c + points_c  # [B, C, N]

#         # add
#         points_attention = points_attention_p + points_attention_c  # [B, C, N]
#         points_attention = points_attention.permute(0, 2, 1)  # [B, N, C]

#         return points_attention


class DenseNet1D(nn.Module):
    def __init__(self, channel_list, seblock=True):
        """
        Input:
            channel_list: a list for input, middle and output data dimension
            seblock: wheather use squeeze-and-excitation networks or not
        """
        super(DenseNet1D, self).__init__()
        self.last_channel = channel_list[2]
        self.conv1 = torch.nn.Conv1d(channel_list[0], channel_list[1], 1)
        self.conv2 = torch.nn.Conv1d(channel_list[0] + channel_list[1], channel_list[1], 1)
        self.conv3 = torch.nn.Conv1d(channel_list[0] + channel_list[1] + channel_list[1], channel_list[1], 1)
        self.conv4 = torch.nn.Conv1d(channel_list[0] + channel_list[1] + channel_list[1] + channel_list[1],
                                     channel_list[2], 1)
        self.bn1 = nn.BatchNorm1d(channel_list[1])
        self.bn2 = nn.BatchNorm1d(channel_list[1])
        self.bn3 = nn.BatchNorm1d(channel_list[1])
        self.bn4 = nn.BatchNorm1d(channel_list[2])
        self.fc1 = nn.Linear(channel_list[2], channel_list[2] // 16)
        self.fc2 = nn.Linear(channel_list[2] // 16, channel_list[2])
        self.bnfc1 = nn.BatchNorm1d(channel_list[2] // 16)
        self.bnfc2 = nn.BatchNorm1d(channel_list[2])
        self.seblock = seblock

    def forward(self, points):
        """
        DenseNet model Conv1D
        input:
            points: [B, D, nsample, npoint]
        return:
            x_dense: [B, D', nsample, npoint]
        """
        x0 = points  # [B, D1, npoint]
        x1 = F.relu(self.bn1(self.conv1(x0)))  # [B, D2, npoint]
        y1 = torch.cat((x0, x1), dim=1)  # [B, D1+D2, npoint]
        x2 = F.relu(self.bn2(self.conv2(y1)))  # [B, D3, npoint]
        y2 = torch.cat((y1, x2), dim=1)  # [B, D1+D2+D3, npoint]
        x3 = F.relu(self.bn3(self.conv3(y2)))  # [B, D4, npoint]
        y3 = torch.cat((y2, x3), dim=1)  # [B, D1+D2+D3+D4, npoint]
        x_dense = F.relu(self.bn4(self.conv4(y3)))  # [B, D5, npoint]

        if self.seblock:
            se = torch.max(x_dense, -1, keepdim=True)[0]  # [B, D', 1]
            se = se.view(-1, self.last_channel)  # [B, D']
            se = F.relu(self.bnfc1(self.fc1(se)))  # [B, D']
            se = torch.sigmoid(self.bnfc2(self.fc2(se)))  # [B, D']
            se = se.view(-1, self.last_channel, 1)  # [B, D', 1]
            x_dense = torch.mul(x_dense, se)  # [B, D', npoint]

        return x_dense


class DenseNet2D(nn.Module):
    def __init__(self, channel_list, seblock=True):
        """
        Input:
            channel_list: a list for input, middle and output data dimension
            seblock: wheather use squeeze-and-excitation networks or not
        """
        super(DenseNet2D, self).__init__()
        self.last_channel = channel_list[2]
        self.conv1 = torch.nn.Conv2d(channel_list[0], channel_list[1], 1)
        self.conv2 = torch.nn.Conv2d(channel_list[0] + channel_list[1], channel_list[1], 1)
        self.conv3 = torch.nn.Conv2d(channel_list[0] + channel_list[1] + channel_list[1], channel_list[1], 1)
        self.conv4 = torch.nn.Conv2d(channel_list[0] + channel_list[1] + channel_list[1] + channel_list[1],
                                     channel_list[2], 1)
        self.bn1 = nn.BatchNorm2d(channel_list[1])
        self.bn2 = nn.BatchNorm2d(channel_list[1])
        self.bn3 = nn.BatchNorm2d(channel_list[1])
        self.bn4 = nn.BatchNorm2d(channel_list[2])
        self.fc1 = nn.Linear(channel_list[2], channel_list[2] // 16)
        self.fc2 = nn.Linear(channel_list[2] // 16, channel_list[2])
        self.bnfc1 = nn.BatchNorm1d(channel_list[2] // 16)
        self.bnfc2 = nn.BatchNorm1d(channel_list[2])
        self.seblock = seblock

    def forward(self, points):
        """
        DenseNet model Conv2D
        input:
            points: [B, D, nsample, npoint]
        return:
            x_dense: [B, D', nsample, npoint]
        """
        x0 = points  # [B, D1, nsample, npoint]
        temp1 = self.conv1(x0)
        temp2 = self.bn1(temp1)
        x1 = F.relu(temp2)
        # x1 = F.relu(self.bn1(self.conv1(x0)))  # [B, D2, nsample, npoint]
        y1 = torch.cat((x0, x1), dim=1)  # [B, D1+D2, nsample, npoint]
        x2 = F.relu(self.bn2(self.conv2(y1)))  # [B, D3, nsample, npoint]
        y2 = torch.cat((y1, x2), dim=1)  # [B, D1+D2+D3, nsample, npoint]
        x3 = F.relu(self.bn3(self.conv3(y2)))  # [B, D4, nsample, npoint]
        y3 = torch.cat((y2, x3), dim=1)  # [B, D1+D2+D3+D4, nsample, npoint]
        x_dense = F.relu(self.bn4(self.conv4(y3)))  # [B, D5, nsample, npoint]

        if self.seblock:
            se = torch.max(x_dense, 2, keepdim=True)[0]  # [B, D', 1, npoint]
            se = torch.max(se, 3, keepdim=True)[0]  # [B, D', 1, 1]
            se = se.view(-1, self.last_channel)  # [B, D']
            se = F.relu(self.bnfc1(self.fc1(se)))  # [B, D']
            se = torch.sigmoid(self.bnfc2(self.fc2(se)))  # [B, D']
            se = se.view(-1, self.last_channel, 1, 1)  # [B, D', 1, 1]
            x_dense = torch.mul(x_dense, se)  # [B, D', nsample, npoint]

        return x_dense


class PNSADenseNet(nn.Module):
    def __init__(self, npoint, radius, nsample, channel_list, normalize_radius=False, group_all=False):
        """
        Input:
            npoint: keyponts number to sample
            radius: sphere radius in a group
            nsample: how many points to group for a sphere
            channel_list: a list for input, middle and output data dimension
            group_all: wheather use group_all or not
            seblock: wheather use squeeze-and-excitation networks or not
        """
        super(PNSADenseNet, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.normalize_radius = normalize_radius
        self.pnsadensesnet = DenseNet2D(channel_list, seblock=True)
        last_channel = channel_list[-1]
        self.DAT = Attention(last_channel)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sub sampled points position data, [B, np, C]
            new_points_res: output points data, [B, np, D']
        """
        if self.group_all:
            new_xyz, new_points, grouped_xyz, fps_points = sample_and_group_all(xyz, points, returnfps=True)
        else:
            new_xyz, new_points, grouped_xyz, fps_points = \
                sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.normalize_radius,
                                 returnfps=True)

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C, nsample, npoint]
        fps_points = fps_points.unsqueeze(3).permute(0, 2, 3, 1)  # [B, C, 1,npoint]
        new_points_graph = new_points - fps_points  # [B, C, nsample,npoint]
        # new_points_graph = self.DAT(new_points_graph)  # [B, C, nsample,npoint]
        new_points_input = torch.cat([new_points_graph, new_points], dim=1)  # [B, 2*(C+D), nsample, npoint]
        new_points_dense = self.pnsadensesnet(new_points_input)  # [B, D', nsample, npoint]
        new_points_dense = torch.max(new_points_dense, 2)[0]  # [B, D', npoint]
        new_points_dense = new_points_dense.permute(0, 2, 1)  # [B, npoint, D']       

        return new_xyz, new_points_dense


class PNFPDenseNet(nn.Module):
    def __init__(self, channel_list):
        """
        Input:
            channel_list: a list for input, middle and output data dimension
        """
        super(PNFPDenseNet, self).__init__()
        self.pnfpdensesnet = DenseNet1D(channel_list, seblock=True)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Interpolate point with densenet
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, N, D]
            points2: input points data, [B, S, D']
        Return:
            new_points_res: upsampled points data, [B, N, D'']
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            assert (S >= 3)
            dists = square_distance(xyz1, xyz2)  # (B, N, S)
            dists, idx = dists.sort(dim=-1)  # (B, N, S)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3], [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  # (B, N, D')

        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, D+D']
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # [B, D+D', N]
        new_points_dense = self.pnfpdensesnet(new_points)  # [B, D'', N]
        new_points_dense = new_points_dense.permute(0, 2, 1)  # [B, N, D'']

        return new_points_dense


def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    # init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i - 1], dims[i]))
        if i == len(dims) - 1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    '''Nxdin ->Nxd1->Nxd2->...-> Nxdout'''

    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


class FoldingNetSingle(nn.Module):
    def __init__(self, dims):
        super(FoldingNetSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)


class FoldingNetDecoder(nn.Module):
    def __init__(self, grid_dims, Folding1_dims, Folding2_dims):
        super(FoldingNetDecoder, self).__init__()
        # Decoder Folding
        # 2D grid: (grid_dims(0) * grid_dims(1)) x 2
        # TODO: normalize the grid to align with the input data
        self.N = grid_dims[0] * grid_dims[1]
        # full 0
        u = (torch.arange(0, grid_dims[0]) / grid_dims[0] - 0.5).repeat(grid_dims[1])  # (grid_dims[1] * grid_dims[0])
        v = (torch.arange(0, grid_dims[1]) / grid_dims[1] - 0.5).expand(grid_dims[0], -1).t().reshape(
            -1)  # (grid_dims[1] * grid_dims[0])
        self.grid = torch.stack((u, v), 1)  # Nx2
        #  1st folding
        self.Fold1 = FoldingNetSingle(Folding1_dims)
        #  2nd folding
        self.Fold2 = FoldingNetSingle(Folding2_dims)

    def forward(self, points):
        points = points.unsqueeze(1)  # Bx1xK
        codeword = points.expand(-1, self.N, -1)  # BxNxK

        # cat 2d grid and feature
        B = codeword.shape[0]  # extract batch size
        tmpGrid = self.grid.cuda()  # Nx2
        tmpGrid = tmpGrid.unsqueeze(0)
        tmpGrid = tmpGrid.expand(B, -1, -1)  # BxNx2

        # 1st folding
        f = torch.cat((tmpGrid, codeword), 2)  # BxNx(K+2)
        f = self.Fold1.forward(f)  # BxNx3

        # 2nd folding
        f = torch.cat((f, codeword), 2)  # BxNx(K+3)
        f = self.Fold2.forward(f)  # BxNx3

        return f


class FoldingNetShapes(nn.Module):
    # add 3 shapes to choose and a learnable layer
    def __init__(self, Folding1_dims, Folding2_dims):
        super(FoldingNetShapes, self).__init__()
        # Decoder Folding
        self.box = make_box()  # 18 * 18 * 6 points
        self.cylinder = make_cylinder()  # same as 1944
        self.sphere = make_sphere()  # 1944 points
        self.grid = torch.Tensor(np.hstack((self.box, self.cylinder, self.sphere)))

        #     1st folding

        self.Fold1 = FoldingNetSingle(Folding1_dims)
        #     2nd folding
        self.Fold2 = FoldingNetSingle(Folding2_dims)
        self.N = 1944  # number of points needed to replicate codeword later; also points in Grid
        self.fc = nn.Linear(9, 9, True)  # geometric transformation

    def forward(self, points):
        points = points.unsqueeze(1)  # Bx1xK
        codeword = points.expand(-1, self.N, -1)  # BxNxK

        # cat 2d grid and feature
        B = codeword.shape[0]  # extract batch size
        tmpGrid = self.grid.cuda()  # Nx9
        tmpGrid = tmpGrid.unsqueeze(0)
        tmpGrid = tmpGrid.expand(B, -1, -1)  # BxNx9
        tmpGrid = self.fc(tmpGrid)  # transform

        # 1st folding
        f = torch.cat((tmpGrid, codeword), 2)  # BxNx(K+9)
        f = self.Fold1.forward(f)  # BxNx3

        # 2nd folding
        f = torch.cat((f, codeword), 2)  # BxNx(K+3)
        f = self.Fold2.forward(f)  # BxNx3

        return f
