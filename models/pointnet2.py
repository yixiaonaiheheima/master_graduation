import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet2_util import PointNetSetAbstraction, PointNetFeaturePropagation


class PointNet2Seg(nn.Module):
    def __init__(self, num_classes, with_rgb=False):
        super(PointNet2Seg, self).__init__()
        self.with_rgb = with_rgb
        if with_rgb:
            additional_channel = 3
        else:
            additional_channel = 0
        self.sa1 = PointNetSetAbstraction(1024, 0.5, 32, 3 + additional_channel, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 1.0, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 2.0, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 4.0, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """

        :param xyz: tensor(b, 3, N)
        :return:
         x: tensor(b, N, num_classes)
         l4_points: tensor(b, 512, 16)
        """
        if self.with_rgb:
            l0_points = xyz[:, 3:, :]  # (b, 3, N)
            l0_xyz = xyz[:, :3, :]  # (b, 3, N)
        else:
            l0_points = None
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # (b, 3, 1024), (b, 64, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (b, 3, 256), (b, 128, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (b, 3, 64), (b, 256, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (b, 3, 16), (b, 512, 16)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (b, 64, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (b, 256, 256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (b, 1024, 128)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (b, 128, N)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))  # (b, 128, N)
        x = self.conv2(x)  # (b, num_classes, N)
        x = F.log_softmax(x, dim=1)  # (b, num_classes, N)
        x = x.permute(0, 2, 1)  # (b, N, num_classes)
        return x, l4_points  # (b, N, num_classes), (b, 512, 16)


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


if __name__ == '__main__':
    import torch
    model = PointNet2Seg(13, with_rgb=True)
    xyz = torch.rand(6, 6, 2048)
    (model(xyz))