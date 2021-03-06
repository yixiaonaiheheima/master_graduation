import torch.nn as nn
import torch
import torch.nn.parallel
import torch.nn.functional as F
from utils.folding_utils import FoldingNetDec
from utils.cross_util import PNSADenseNet, PNFPDenseNet
# from utils.pointnet_util import PNSADenseNet, PNFPDenseNet


class PointSemantic(nn.Module):
    def __init__(self, num_classes, with_rgb=False, addition_channel=0):
        super(PointSemantic, self).__init__()
        self.with_rgb = with_rgb
        self.feature_channel = addition_channel
        if with_rgb:
            self.feature_channel += 3
        print('feature_channel is %d' % self.feature_channel)

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
