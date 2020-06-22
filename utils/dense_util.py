import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.basics_util import square_distance, index_points, sample_and_group, sample_and_group_all, \
    make_box, make_sphere, make_cylinder


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=1, stride=1, padding=0,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input):
    #     # type: (List[Tensor]) -> (Tensor)
    #     pass
    #
    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input):
    #     # type: (Tensor) -> (Tensor)
    #     pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class _DenseBlock(nn.ModuleDict):
    # _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class myDenseBlock(nn.Module):
    def __init__(self, num_features, num_layers, growth_rate=32,
                 bn_size=4, drop_rate=0, memory_efficient=False, transition=True):
        super(myDenseBlock, self).__init__()
        block = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.features = nn.Sequential()
        self.features.add_module('denseblock', block)
        num_features = num_features + num_layers * growth_rate
        if transition:
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features // 2)
            self.features.add_module('transition', trans)
            num_features = num_features // 2

    def forward(self, x):
        features = self.features(x)
        return features


class PNSADenseNet(nn.Module):
    def __init__(self, npoint, radius, nsample, num_features, num_layers, normalize_radius=False, group_all=False):
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
        self.pnsadensesnet = myDenseBlock(num_features, num_layers)

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
        new_points_input = torch.cat([new_points_graph, new_points], dim=1)  # [B, 2*(C+D), nsample, npoint]
        new_points_dense = self.pnsadensesnet(new_points_input)  # [B, D', nsample, npoint]
        new_points_dense = torch.max(new_points_dense, 2)[0]  # [B, D', npoint]
        new_points_dense = new_points_dense.permute(0, 2, 1)  # [B, npoint, D']

        return new_xyz, new_points_dense


class _DenseLayer1D(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer1D, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate,
                                           kernel_size=1, stride=1, padding=0,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input):
    #     # type: (List[Tensor]) -> (Tensor)
    #     pass
    #
    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input):
    #     # type: (Tensor) -> (Tensor)
    #     pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _Transition1D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition1D, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class _DenseBlock1D(nn.ModuleDict):
    # _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock1D, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer1D(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class myDenseBlock1D(nn.Module):
    def __init__(self, num_features, num_layers, growth_rate=32,
                 bn_size=4, drop_rate=0, memory_efficient=False, transition=True):
        super(myDenseBlock1D, self).__init__()
        block = _DenseBlock1D(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.features = nn.Sequential()
        self.features.add_module('denseblock', block)
        num_features = num_features + num_layers * growth_rate
        if transition:
            trans = _Transition1D(num_input_features=num_features,
                                num_output_features=num_features // 2)
            self.features.add_module('transition', trans)
            num_features = num_features // 2

    def forward(self, x):
        features = self.features(x)
        return features


class PNFPDenseNet(nn.Module):
    def __init__(self, num_features, num_layers):
        """
        Input:
            channel_list: a list for input, middle and output data dimension
        """
        super(PNFPDenseNet, self).__init__()
        self.pnfpdensesnet = myDenseBlock1D(num_features, num_layers)

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