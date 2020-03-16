import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def get_activation(argument):
    getter = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "softplus": F.softplus,
        "logsigmoid": F.logsigmoid,
        "softsign": F.softsign,
        "tanh": F.tanh,
    }
    return getter.get(argument, "Invalid activation")


class PointNet(nn.Module):
    def __init__(self, nlatent=1024, dim_input=3):
        """
        PointNet Encoder
        See : PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
                Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
        """

        super(PointNet, self).__init__()
        self.dim_input = dim_input
        self.conv1 = torch.nn.Conv1d(dim_input, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)

        self.nlatent = nlatent

    def forward(self, x):
        """

        :param x: tensor(B, 3, N)
        :return: tensor(B, 1024)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # (B, 1024, N)
        x, _ = torch.max(x, 2)  # (B, 1024)
        x = x.view(-1, self.nlatent)  # (B, 1024)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)


class Mapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        self.opt = opt
        self.bottleneck_size = opt.bottleneck_size
        self.input_size = opt.dim_template
        self.dim_output = 3
        self.hidden_neurons = opt.hidden_neurons
        self.num_layers = opt.num_layers
        super(Mapping2Dto3D, self).__init__()
        print(
            f"New MLP decoder : hidden size {opt.hidden_neurons}, num_layers {opt.num_layers}, activation {opt.activation}")

        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)

        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])

        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_neurons)

        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        self.activation = get_activation(opt.activation)

    def forward(self, x, latent):
        """
        x: tensor(1, 3, nb_pts_in_primitive)
        lantent: tensor(B, bottleneck_size, 1)
        return: tensor(B, 3,  nb_pts_in_primitive)
        """
        x = self.conv1(x) + latent  # (B, bottleneck_size, nb_pts_in_primitive)
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.opt.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        return self.last_conv(x)


def get_template(template_type, device=0):
    getter = {
        "SQUARE": SquareTemplate,
        "SPHERE": SphereTemplate,
    }
    template = getter.get(template_type, "Invalid template")
    return template(device=device)


class Template(object):
    @staticmethod
    def get_random_points(self):
        print("Please implement get_random_points ")


class SphereTemplate(Template):
    def __init__(self, device=0, grain=6):
        self.device = device
        self.dim = 3
        self.npoints = 0

    @staticmethod
    def get_random_points(shape, device="cuda"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 3, x ... x]
        """
        assert shape[1] == 3, "shape should have 3 in dim 1"
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()  # (1, 3, N)
        rand_grid.data.normal_(0, 1)
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))  # (1, 3, N)
        return Variable(rand_grid)


class SquareTemplate(Template):
    def __init__(self, device=0):
        self.device = device
        self.dim = 2
        self.npoints = 0

    @staticmethod
    def get_random_points(shape, device="cuda"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 2, x ... x]
        """
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.uniform_(0, 1)
        return Variable(rand_grid)

    @staticmethod
    def generate_square(grain):
        """
        Generate a square mesh from a regular grid.
        :param grain:
        :return:
        """
        grain = int(grain)
        grain = grain - 1  # to return grain*grain points
        # generate regular grid
        faces = []
        vertices = []
        for i in range(0, int(grain + 1)):
            for j in range(0, int(grain + 1)):
                vertices.append([i / grain, j / grain, 0])

        for i in range(1, int(grain + 1)):
            for j in range(0, (int(grain + 1) - 1)):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i + 1,
                              j + (grain + 1) * (i - 1)])
        for i in range(0, (int((grain + 1)) - 1)):
            for j in range(1, int((grain + 1))):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i - 1,
                              j + (grain + 1) * (i + 1)])

        return np.array(vertices), np.array(faces)