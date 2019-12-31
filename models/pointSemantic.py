import torch.nn as nn
import torch
import torch.nn.parallel
import torch.nn.functional as F
from utils.folding_utils import FoldingNetDec
from utils.pointnet_util import PNSADenseNet, PNFPDenseNet


class PointSemantic_cross(nn.Module):
    def __init__(self, num_classes, with_rgb=False, addition_channel=0):
        super(PointSemantic_cross, self).__init__()
        self.with_rgb = with_rgb
        self.feature_channel = addition_channel
        if with_rgb:
            self.feature_channel += 3
        # Encoder
        # we discard the default setting radius [0.5,1.0,2.0,4.0] as we don't normalize point cloud before network
        self.sa1 = PNSADenseNet(1024, 0.5, 32, [2 * (self.feature_channel + 3), 64, 64], False, False)
        self.sa2 = PNSADenseNet(256, 1.0, 32, [2 * (64 + 3), 128, 128], False, False)
        self.sa3 = PNSADenseNet(64, 2.0, 32, [2 * (128 + 3), 256, 256], False, False)
        self.sa4 = PNSADenseNet(16, 4.0, 32, [2 * (256 + 3), 512, 512], False, False)

        # Decoder-semantic3d
        self.fp4 = PNFPDenseNet([768, 256, 256])
        self.fp3 = PNFPDenseNet([384, 256, 256])
        self.fp2 = PNFPDenseNet([320, 256, 128])
        self.fp1 = PNFPDenseNet([128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        # Decoder-kitti
        self.fold = FoldingNetDec(90)

    def forward(self, pointcloud1, pointcloud2):
        """
        Input:
             pointcloud1: input tensor anchor (B, N, C) semantic_3d, C is 3[+3][+7]
             pointcloud2: input tensor anchor (B, N, C) npm3d
        return:
             lp: absolute position (2B, 3)
             lx: relative translation (B, 3)
        """
        # input point cloud
        assert(pointcloud1.shape == pointcloud2.shape)
        if self.feature_channel != 0:
            l0_xyz1 = pointcloud1[:, :, :3]  # (b, N, 3)
            l0_xyz2 = pointcloud2[:, :, :3]  # (b, N, 3)
            l0_points1 = pointcloud1[:, :, 3:]  # (b, N, f)
            l0_points2 = pointcloud2[:, :, 3:]  # (b, N, f)
        else:
            l0_xyz1 = pointcloud1
            l0_xyz2 = pointcloud2
            l0_points1 = None
            l0_points2 = None

        # semantic3d encoder (share weighted)
        l1_xyz1, l1_points1 = self.sa1(l0_xyz1, l0_points1)  # (b, 1024, 3), (b, 1024, 64)
        l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)  # (b, 256, 3), (b, 256, 128)
        l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)  # (b, 64, 3), (b, 64, 256)
        l4_xyz1, l4_points1 = self.sa4(l3_xyz1, l3_points1)  # (b, 16, 3), (b, 16, 512)

        # npm3d encoder (share weighted)
        l1_xyz2, l1_points2 = self.sa1(l0_xyz2, l0_points2)  # (b, 1024, 3), (b, 1024, 64)
        l2_xyz2, l2_points2 = self.sa2(l1_xyz2, l1_points2)  # (b, 256, 3), (b, 256, 128)
        l3_xyz2, l3_points2 = self.sa3(l2_xyz2, l2_points2)  # (b, 64, 3), (b, 64, 256)
        l4_xyz2, l4_points2 = self.sa4(l3_xyz2, l3_points2)  # (b, 16, 3), (b, 16, 512)

        # semantic3d decoder
        l3_points1 = self.fp4(l3_xyz1, l4_xyz1, l3_points1, l4_points1)  # (b, 64, 256)
        l2_points1 = self.fp3(l2_xyz1, l3_xyz1, l2_points1, l3_points1)  # (b, 256, 256)
        l1_points1 = self.fp2(l1_xyz1, l2_xyz1, l1_points1, l2_points1)  # (b, 1024, 128)
        l0_points1 = self.fp1(l0_xyz1, l1_xyz1, None, l1_points1)  # (b, N, 128)
        l0_points1 = l0_points1.permute(0, 2, 1)  # (b, 128, N)
        semantic3d_points = self.drop1(F.relu(self.bn1(self.conv1(l0_points1))))  # (b, 128, N)
        semantic3d_points = self.conv2(semantic3d_points)  # (b, num_classes, N)
        semantic3d_prob = F.log_softmax(semantic3d_points, dim=1)  # (b, num_classes, N)
        semantic3d_prob = semantic3d_prob.permute(0, 2, 1)  # (b, N, num_classes)

        # npm3d decoder (folding)
        global_feature, _ = torch.max(l4_points2, dim=1)  # (b, 512)
        npm_pc_reconstructed = self.fold(global_feature)  # (b, 3, 90^2)
        npm_pc_reconstructed = npm_pc_reconstructed.permute(0, 2, 1)  # (b, 90^2, 3)

        return semantic3d_prob, npm_pc_reconstructed


class PointSemantic(nn.Module):
    def __init__(self, num_classes, with_rgb=False, addition_channel=0):
        super(PointSemantic, self).__init__()
        self.with_rgb = with_rgb
        self.feature_channel = addition_channel
        if with_rgb:
            self.feature_channel += 3

        # Encoder
        # we discard the default setting radius [0.5,1.0,2.0,4.0] as we don't normalize point cloud before network
        self.sa1 = PNSADenseNet(1024, 0.5, 32, [2 * (self.feature_channel + 3), 64, 64], False, False)
        self.sa2 = PNSADenseNet(256, 1.0, 32, [2 * (64 + 3), 128, 128], False, False)
        self.sa3 = PNSADenseNet(64, 2.0, 32, [2 * (128 + 3), 256, 256], False, False)
        self.sa4 = PNSADenseNet(16, 4.0, 32, [2 * (256 + 3), 512, 512], False, False)

        # Decoder-semantic3d
        self.fp4 = PNFPDenseNet([768, 256, 256])
        self.fp3 = PNFPDenseNet([384, 256, 256])
        self.fp2 = PNFPDenseNet([320, 256, 128])
        self.fp1 = PNFPDenseNet([128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, pc, return_embed=False):
        """

        :param pc: tensor(b, N, 3+f)
        :param return_embed: bool, return embedding for visualization
        :return:
        """
        # input point cloud
        if self.feature_channel != 0:
            l0_xyz1 = pc[:, :, :3]  # (b, N, 3)
            l0_points1 = pc[:, :, 3:]  # (b, N, f)
        else:
            l0_xyz1 = pc
            l0_points1 = None

        # semantic3d encoder (share weighted)
        l1_xyz1, l1_points1 = self.sa1(l0_xyz1, l0_points1)  # (b, 1024, 3), (b, 1024, 64)
        l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)  # (b, 256, 3), (b, 256, 128)
        l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)  # (b, 64, 3), (b, 64, 256)
        l4_xyz1, l4_points1 = self.sa4(l3_xyz1, l3_points1)  # (b, 16, 3), (b, 16, 512)

        # semantic3d decoder
        l3_points1 = self.fp4(l3_xyz1, l4_xyz1, l3_points1, l4_points1)  # (b, 64, 256)
        l2_points1 = self.fp3(l2_xyz1, l3_xyz1, l2_points1, l3_points1)  # (b, 256, 256)
        l1_points1 = self.fp2(l1_xyz1, l2_xyz1, l1_points1, l2_points1)  # (b, 1024, 128)
        l0_points1 = self.fp1(l0_xyz1, l1_xyz1, None, l1_points1)  # (b, N, 128)
        l0_points1 = l0_points1.permute(0, 2, 1)  # (b, 128, N)
        semantic3d_points = self.drop1(F.relu(self.bn1(self.conv1(l0_points1))))  # (b, 128, N)
        semantic3d_points = self.conv2(semantic3d_points)  # (b, num_classes, N)
        semantic3d_prob = F.log_softmax(semantic3d_points, dim=1)  # (b, num_classes, N)
        semantic3d_prob = semantic3d_prob.permute(0, 2, 1)  # (b, N, num_classes)

        if return_embed:
            return semantic3d_prob, l0_points1.permute(0, 2, 1)
        else:
            return semantic3d_prob


if __name__ == '__main__':
    pass
