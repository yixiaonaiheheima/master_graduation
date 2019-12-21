import torch.nn as nn
import torch
import torch.nn.parallel
import torch.nn.functional as F
from utils.pointnet_util import PNSADenseNet_GAT, PNSADenseNet_Dual, PNFPDenseNet, FoldingNetDecoder, FoldingNetShapes


class PointSemantic(nn.Module):
    def __init__(self, num_classes):
        super(PointSemantic, self).__init__()
        # Encoder-GAT
        self.sa1 = PNSADenseNet_GAT(1024, 0.5, 64, [3 + 3, 32, 64], 0.6, 0.2, False, False)
        self.sa2 = PNSADenseNet_GAT(256, 1, 48, [64 + 3, 64, 128], 0.6, 0.2, False, False)
        self.sa3 = PNSADenseNet_GAT(64, 2, 32, [128 + 3, 128, 256], 0.6, 0.2, False, False)
        self.sa4 = PNSADenseNet_GAT(16, 4, 16, [256 + 3, 256, 512], 0.6, 0.2, False, False)

        # Encoder-Dual attention
        # self.sa1 = PNSADenseNet_Dual(1024, 0.5, 64, [3 + 3, 32, 64], False, True, False)
        # self.sa2 = PNSADenseNet_Dual(256, 1, 48, [64 + 3, 64, 128], False, True, False)
        # self.sa3 = PNSADenseNet_Dual(64, 2, 32, [128 + 3, 128, 256], False, True, False)
        # self.sa4 = PNSADenseNet_Dual(16, 4, 16, [256 + 3, 256, 512], False, True, False)

        # Decoder-semantic3d
        self.fp1 = PNFPDenseNet([768, 256, 256])
        self.fp2 = PNFPDenseNet([384, 256, 256])
        self.fp3 = PNFPDenseNet([320, 256, 128])
        self.fp_semantic3d = PNFPDenseNet([128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        # Decoder-kitti
        self.fp_kitti = PNFPDenseNet([128, 64, 3])
        self.fold = FoldingNetDecoder([2, 2], [521, 512, 512, 3], [521, 512, 512, 3])

    def forward(self, pointcloud1, pointcloud2):
        """
        Input:
             pointcloud1: input tensor anchor (B, N, 6) semantic_3d
             pointcloud2: input tensor anchor (B, N, 6) kitti
        return:
             lp: absolute position (2B, 3)
             lx: relative translation (B, 3)
        """
        # input point cloud
        assert(pointcloud1.shape == pointcloud2.shape)
        l0_xyz1 = pointcloud1[:, :, :3]
        l0_xyz2 = pointcloud2[:, :, :3]
        if pointcloud1.shape[2] <= 3:
            l0_points1 = None
            l0_points2 = None
        else:
            l0_points1 = pointcloud1[:, :, 3:]
            l0_points2 = pointcloud2[:, :, 3:]

        # semantic3d encoder (share weighted)
        l1_xyz1, l1_points1 = self.sa1(l0_xyz1, l0_points1)  # (b, 1024, 3), (b, 1024, 24, 64)
        l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)  # (b, 256, 3), (b, 256, 46, 128)
        l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)  # (b, 64, 3), (b, 64, 32, 256)
        l4_xyz1, l4_points1 = self.sa4(l3_xyz1, l3_points1)  # (b, 16, 3), (b, 16, 15, 512)

        # kitti encoder (share weighted)
        l1_xyz2, l1_points2 = self.sa1(l0_xyz2, l0_points2)  # (b, 1024, 3), (b, 1024, 24, 64)
        l2_xyz2, l2_points2 = self.sa2(l1_xyz2, l1_points2)  # (b, 256, 3), (b, 256, 46, 128)
        l3_xyz2, l3_points2 = self.sa3(l2_xyz2, l2_points2)  # (b, 64, 3), (b, 64, 32, 256)
        l4_xyz2, l4_points2 = self.sa4(l3_xyz2, l3_points2)  # (b, 16, 3), (b, 16, 15, 512)

        # semantic3d decoder
        l3_points1 = self.fp4(l3_xyz1, l4_xyz1, l3_points1, l4_points1)
        l2_points1 = self.fp3(l2_xyz1, l3_xyz1, l2_points1, l3_points1)
        l1_points1 = self.fp2(l1_xyz1, l2_xyz1, l1_points1, l2_points1)
        l0_points1 = self.fp1(l0_xyz1, l1_xyz1, l0_points1, l1_points1)  # (b, N, 128)
        l0_points1 = l0_points1.permute(0, 2, 1)  # (b, 128, N)
        semantic3d_x = self.drop1(F.relu(self.bn1(self.conv1(l0_points1))))  # (b, 128, N)
        semantic3d_x = self.conv2(semantic3d_x)  # (b, num_classes, N)
        semantic3d_x = F.log_softmax(semantic3d_x, dim=1)  # (b, num_classes, N)
        semantic3d_x = semantic3d_x.permute(0, 2, 1)  # (b, N, num_classes)

        # kitti decoder (interpolated_points)
        l3_points2 = self.fp4(l3_xyz2, l4_xyz2, l3_points2, l4_points2)
        l2_points2 = self.fp3(l2_xyz2, l3_xyz2, l2_points2, l3_points2)
        l1_points2 = self.fp2(l1_xyz2, l2_xyz2, l1_points2, l2_points2)
        kitti_points = self.fp1(l0_xyz2, l1_xyz2, l0_points2, l1_points2)

        # kitti decoder (folding)
        # kitti_points = self.fold(l4_points2)

        return semantic3d_x, kitti_points


if __name__ == '__main__':
    model = PointSemantic(8)
