import torch.nn as nn
import torch
import torch.nn.parallel
import torch.nn.functional as F
from utils.folding_utils import FoldingNetDec
# from utils.cross_util import PNSADenseNet, PNFPDenseNet
from utils.pointnet_util import PNSADenseNet, PNFPDenseNet


class PointSemantic_cross(nn.Module):
    def __init__(self, num_classes, with_rgb=False, addition_channel=0):
        super(PointSemantic_cross, self).__init__()
        self.with_rgb = with_rgb
        self.feature_channel = addition_channel
        if with_rgb:
            self.feature_channel += 3
        print("feature_channel in PointSemantic_cross is %d" % self.feature_channel)
        # Encoder
        self.conv0 = nn.Conv1d(self.feature_channel + 3, 32, 1)
        self.bn0 = nn.BatchNorm1d(32)
        # we discard the default setting radius [0.5,1.0,2.0,4.0] as we don't normalize point cloud before network
        self.sa1 = PNSADenseNet(1024, 0.5, 32, [2 * (32 + 3), 64, 64], False, False)
        self.sa2 = PNSADenseNet(256, 1.0, 32, [2 * (64 + 3), 128, 128], False, False)
        self.sa3 = PNSADenseNet(64, 2.0, 16, [2 * (128 + 3), 256, 256], False, False)
        self.sa4 = PNSADenseNet(16, 4.0, 16, [2 * (256 + 3), 512, 512], False, False)

        # self.fa2 = PNSADenseNet(256, 1.0, 32, [2 * (64 + 3), 128, 128], False, False)
        # self.fa3 = PNSADenseNet(64, 2.0, 16, [2 * (128 + 3), 256, 256], False, False)
        # self.fa4 = PNSADenseNet(16, 4.0, 16, [2 * (256 + 3), 512, 512], False, False)
        # self.fa5 = PNSADenseNet(None, None, None, [2 * (512 + 3), 512, 512], False, True)
        self.conv1 = nn.Conv1d(32, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        # Decoder-semantic3d
        self.fp4 = PNFPDenseNet([768, 256, 256])
        self.fp3 = PNFPDenseNet([384, 256, 256])
        self.fp2 = PNFPDenseNet([320, 256, 128])
        self.fp1 = PNFPDenseNet([128, 128, 128])

        # seg-semantic3d
        self.convs1 = nn.Conv1d(128, 128, 1)
        self.bns1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(128, num_classes, 1)

        # reconstruction-npm3d
        self.fold = FoldingNetDec(64)
        # self.fc1 = nn.Linear(512, 1024)
        # self.bnfc1 = nn.BatchNorm1d(1024)
        # self.fc2 = nn.Linear(1024, 2048)
        # self.bnfc2 = nn.BatchNorm1d(2048)
        # self.fc3 = nn.Linear(2048, 4096)
        # self.bnfc3 = nn.BatchNorm1d(4096)
        # self.fc4 = nn.Linear(4096, 4096*3)
        # self.th = nn.Tanh()

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

        l0_points1 = l0_xyz1.permute(0, 2, 1)  # (B, 3, N)
        l0_points2 = l0_xyz2.permute(0, 2, 1)  # (B, 3, N)
        l0_points1 = F.relu(self.bn0(self.conv0(l0_points1)))
        l0_points2 = F.relu(self.bn0(self.conv0(l0_points2)))
        l0_points1 = l0_points1.permute(0, 2, 1)
        # semantic3d encoder (share weighted)
        l1_xyz1, l1_points1 = self.sa1(l0_xyz1, l0_points1)  # (b, 1024, 3), (b, 1024, 64)
        l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)  # (b, 256, 3), (b, 256, 128)
        l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)  # (b, 64, 3), (b, 64, 256)
        l4_xyz1, l4_points1 = self.sa4(l3_xyz1, l3_points1)  # (b, 16, 3), (b, 16, 512)

        # npm3d encoder (share weighted)
        # l1_xyz2, l1_points2 = self.sa1(l0_xyz2, l0_points2)  # (b, 1024, 3), (b, 1024, 64)
        # l2_xyz2, l2_points2 = self.fa2(l1_xyz2, l1_points2)  # (b, 256, 3), (b, 256, 128)
        # l3_xyz2, l3_points2 = self.fa3(l2_xyz2, l2_points2)  # (b, 64, 3), (b, 64, 256)
        # l4_xyz2, l4_points2 = self.fa4(l3_xyz2, l3_points2)  # (b, 16, 3), (b, 16, 512)
        # l5_xyz2, l5_points2 = self.fa5(l4_xyz2, l4_points2)  # (b, 1, 3), (b, 1, 512)
        l1_points2 = F.relu(self.bn1(self.conv1(l0_points2)))
        l2_points2 = F.relu(self.bn2(self.conv2(l1_points2)))
        l3_points2 = F.relu(self.bn3(self.conv3(l2_points2)))
        l4_points2 = F.relu(self.bn4(self.conv4(l3_points2)))

        # semantic3d decoder
        l3_points1 = self.fp4(l3_xyz1, l4_xyz1, l3_points1, l4_points1)  # (b, 64, 256)
        l2_points1 = self.fp3(l2_xyz1, l3_xyz1, l2_points1, l3_points1)  # (b, 256, 256)
        l1_points1 = self.fp2(l1_xyz1, l2_xyz1, l1_points1, l2_points1)  # (b, 1024, 128)
        l0_points1 = self.fp1(l0_xyz1, l1_xyz1, None, l1_points1)  # (b, N, 128)
        l0_points1 = l0_points1.permute(0, 2, 1)  # (b, 128, N)
        semantic3d_points = self.drop1(F.relu(self.bns1(self.convs1(l0_points1))))  # (b, 128, N)
        semantic3d_points = self.convs2(semantic3d_points)  # (b, num_classes, N)
        semantic3d_prob = F.log_softmax(semantic3d_points, dim=1)  # (b, num_classes, N)
        semantic3d_prob = semantic3d_prob.permute(0, 2, 1)  # (b, N, num_classes)

        # npm3d decoder (folding)
        global_feature, _ = torch.max(l4_points2, dim=-1)  # (b, 512)
        # global_feature = global_feature.squeeze(1)
        npm_pc_reconstructed = self.fold(global_feature)  # (b, 3, 90^2)
        # net = F.relu(self.bnfc1(self.fc1(global_feature)))
        # net = F.relu(self.bnfc2(self.fc2(net)))
        # net = F.relu(self.bnfc3(self.fc3(net)))
        # net = self.th(self.fc4(net))
        # batchsize = net.size()[0]
        # npm_pc_reconstructed = net.view(batchsize, 4096, 3)
        npm_pc_reconstructed = npm_pc_reconstructed.permute(0, 2, 1)  # (b, 90^2, 3)

        return semantic3d_prob, npm_pc_reconstructed