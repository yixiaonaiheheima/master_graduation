"""
Author: Austin J. Garrett

PyTorch implementation of the PointCNN paper, as specified in:
  https://arxiv.org/pdf/1801.07791.pdf
Original paper by: Yangyan Li, Rui Bu, Mingchao Sun, Baoquan Chen
"""

# External Modules
import torch
import torch.nn as nn
from torch import FloatTensor
import numpy as np
from typing import Tuple, Callable, Optional
import torch.nn.functional as F

# Internal Modules
from .util_funcs import UFloatTensor, ULongTensor, knn_indices_func_gpu
from .util_layers import Conv, SepConv, Dense, EndChannels
from utils.basics_util import farthest_point_sample, index_points


class XConv(nn.Module):
    """ Convolution over a single point and its neighbors.  """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int,
                 P: int, C_mid: int, depth_multiplier: int) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points.
        :param C_mid: Dimensionality of lifted point features.
        :param depth_multiplier: Depth multiplier for internal depthwise separable convolution.
        """
        super(XConv, self).__init__()

        # if __debug__:
        self.C_in = C_in
        self.C_mid = C_mid
        self.dims = dims
        self.K = K

        self.P = P

        # Additional processing layers
        # self.pts_layernorm = LayerNorm(2, momentum = 0.9)

        # Main dense linear layers
        self.dense1 = Dense(dims, C_mid)
        self.dense2 = Dense(C_mid, C_mid)

        # Layers to generate X
        self.x_trans = nn.Sequential(
            EndChannels(Conv(
                in_channels=dims,
                out_channels=K*K,
                kernel_size=(1, K),
                with_bn=False
            )),
            Dense(K*K, K*K, with_bn=False),
            Dense(K*K, K*K, with_bn=False, activation=None)
        )
        
        self.end_conv = EndChannels(SepConv(
            in_channels=C_mid + C_in,
            out_channels=C_out,
            kernel_size=(1, K),
            depth_multiplier=depth_multiplier
        )).cuda()
        
    def forward(self, x: Tuple[UFloatTensor,            # (N, P, dims)
                               UFloatTensor,            # (N, P, K, dims)
                               Optional[UFloatTensor]]  # (N, P, K, C_in)
                ) -> UFloatTensor:                      # (N, P, C_out)
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the feature
          associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x

        if fts is not None:
            assert(rep_pt.size()[0] == pts.size()[0] == fts.size()[0])  # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1] == fts.size()[1])  # Check P is equal.
            assert(pts.size()[2] == fts.size()[2] == self.K)            # Check K is equal.
            assert(fts.size()[3] == self.C_in)                          # Check C_in is equal.
        else:
            assert(rep_pt.size()[0] == pts.size()[0])                   # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1])                   # Check P is equal.
            assert(pts.size()[2] == self.K)                             # Check K is equal.
        assert(rep_pt.size()[2] == pts.size()[3] == self.dims)          # Check dims is equal.

        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = torch.unsqueeze(rep_pt, dim=2)  # (N, P, 1, dims)

        # Move pts to local coordinate system of rep_pt.
        pts_local = pts - p_center  # (N, P, K, dims)
        # pts_local = self.pts_layernorm(pts - p_center)

        # Individually lift each point into C_mid space.
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted = self.dense2(fts_lifted0)  # (N, P, K, C_mid)

        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = torch.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)

        # Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, P, self.K, self.K)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)

        # Weight and permute fts_cat with the learned X.
        fts_X = torch.matmul(X, fts_cat)  # (N, P, K, C_mid + C_in)
        fts_p = self.end_conv(fts_X).squeeze(dim=2)  # (N, P, C_out)
        return fts_p


class PointCNN(nn.Module):
    """ Pointwise convolutional model. """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int, D: int, P: int,
                 r_indices_func: Callable[[UFloatTensor,  # (N, P, dims)
                                           UFloatTensor,  # (N, x, dims)
                                           int, int],     # K, D
                                          ULongTensor]    # (N, P, K)
                 ) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param D: "Spread" of neighboring points.
        :param P: Number of representative points.
        :param r_indices_func: Selector function of the type,
          INPUTS
          rep_pts : Representative points.
          pts  : Point cloud.
          K : Number of points for each region.
          D : "Spread" of neighboring points.

          OUTPUT
          pts_idx : Array of indices into pts such that pts[pts_idx] is the set
          of points in the "region" around rep_pt.
        """
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4

        if C_in == 0:
            depth_multiplier = 1
        else:
            depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)

        self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        self.dense = Dense(C_in, C_out // 2) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.D = D

    @staticmethod
    def select_region(pts: UFloatTensor,  # (N, x, dims)
                      pts_idx: ULongTensor      # (N, P, K)
                      ) -> UFloatTensor:        # (N, P, K, dims)
        """
        Selects neighborhood points based on output of r_indices_func.
        :param pts: Point cloud to select regional points from.
        :param pts_idx: Indices of points in region to be selected.
        :return: Local neighborhoods around each representative point.
        """
        regions = torch.stack([
            pts[n][idx, :] for n, idx in enumerate(torch.unbind(pts_idx, dim=0))
        ], dim=0)
        return regions

    def forward(self, x: Tuple[FloatTensor,  # (N, P, dims)
                               FloatTensor,  # (N, x, dims)
                               FloatTensor]  # (N, x, C_in)
                ) -> FloatTensor:            # (N, P, C_out)
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :param x: (rep_pts, pts, fts) where
          - rep_pts: Representative points.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the
          feature associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated to rep_pts.
        """
        rep_pts, pts, fts = x
        fts = self.dense(fts) if fts is not None else fts

        # This step takes ~97% of the time. Prime target for optimization: KNN on GPU.
        pts_idx = self.r_indices_func(rep_pts.cpu(), pts.cpu()).cuda()  # (N, P, K)
        # -------------------------------------------------------------------------- #

        pts_regional = PointCNN.select_region(pts, pts_idx)  # (N, P, K, dims)
        fts_regional = PointCNN.select_region(fts, pts_idx) if fts is not None else fts  # (N, P, K, C_in)
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))  # (N, P, C_out)

        return fts_p


class RandPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int, D: int, P: int,
                 r_indices_func: Callable[[UFloatTensor,  # (N, P, dims)
                                           UFloatTensor,  # (N, x, dims)
                                           int, int],
                                          ULongTensor]    # (N, P, K)
                 ) -> None:
        """ See documentation for PointCNN. """
        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.P = P

    def forward(self, x: Tuple[UFloatTensor,  # (N, x, dims)
                               UFloatTensor]  # (N, x, C_in)
                ) -> Tuple[UFloatTensor,        # (N, P, dims)
                           UFloatTensor]:       # (N, P, C_out)
        """
        Given a point cloud, and its corresponding features, return a new set
        of randomly-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].
        :return: Randomly subsampled points and their features.
        """
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            # Select random set of indices of subsampled points.
            idx = np.random.choice(pts.size()[1], self.P, replace=False).tolist()
            rep_pts = pts[:, idx, :]  # (N, P, dims)
        else:
            # All input points are representative points.
            rep_pts = pts
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        return rep_pts, rep_pts_fts


class FPSPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int, D: int, P: int,
                 r_indices_func: Callable[[UFloatTensor,  # (N, P, dims)
                                           UFloatTensor,  # (N, x, dims)
                                           int, int],
                                          ULongTensor]    # (N, P, K)
                 ) -> None:
        """ See documentation for PointCNN. """
        super(FPSPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.P = P

    def forward(self, x: Tuple[UFloatTensor,  # (N, x, dims)
                               UFloatTensor]  # (N, x, C_in)
                ) -> Tuple[UFloatTensor,        # (N, P, dims)
                           UFloatTensor]:       # (N, P, C_out)
        """
        Given a point cloud, and its corresponding features, return a new set
        of fartest-points-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].
        :return: Randomly subsampled points and their features.
        """
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            # Select random set of indices of subsampled points.
            fps_idx = farthest_point_sample(pts, self.P)  # [N, P]
            rep_pts = index_points(pts, fps_idx)  # [N, P, dim]
        else:
            # All input points are representative points.
            rep_pts = pts
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))  # [N, P, C_in]
        return rep_pts, rep_pts_fts


class Xdconv(nn.Module):
    """ Pointwise convolutional model. """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int, D: int, P: int,
                 r_indices_func: Callable[[UFloatTensor,  # (N, P, dims)
                                           UFloatTensor,  # (N, x, dims)
                                           int, int],
                                          ULongTensor]  # (N, P, K)
                 ) -> None:
        super(Xdconv, self).__init__()
        C_mid = C_in // 4
        depth_multiplier = 1
        self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        self.x_conv = XConv(C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.D = D
        self.dense = Dense(2*C_out, C_out)

    def forward(self, x: Tuple[FloatTensor,  # (N, P, dims)
                               FloatTensor,  # (N, P, C_out)
                               FloatTensor,  # (N, x, dims)
                               FloatTensor]  # (N, x, C_in)
                ) -> FloatTensor:            # (N, P, C_out)
        rep_pts, rep_fts, pts, fts = x
        pts_idx = self.r_indices_func(rep_pts.cpu(), pts.cpu()).cuda()  # (N, P, K)
        pts_regional = PointCNN.select_region(pts, pts_idx)  # (N, P, K, dims)
        fts_regional = PointCNN.select_region(fts, pts_idx) if fts is not None else fts  # (N, P, K, C_in)

        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))  # (N, P, C_out)
        fts_concat = torch.cat([fts_p, rep_fts], dim=-1)  # (N, P, C_out*2)
        fts_fuse = self.dense(fts_concat)  # (N, P, C_out)

        return fts_fuse


class PointCNN_seg(nn.Module):
    def __init__(self, num_classes, feature_dim=3):
        super(PointCNN_seg, self).__init__()
        x = 8
        self.pointcnn1 = FPSPointCNN(C_in=feature_dim, C_out=16*x, dims=3, K=12, D=1, P=-1, r_indices_func=knn_indices_func_gpu)
        self.pointcnn2 = FPSPointCNN(C_in=16*x, C_out=32*x, dims=3, K=16, D=1, P=768, r_indices_func=knn_indices_func_gpu)
        self.pointcnn3 = FPSPointCNN(C_in=32*x, C_out=64*x, dims=3, K=16, D=2, P=384, r_indices_func=knn_indices_func_gpu)
        self.pointcnn4 = FPSPointCNN(C_in=64*x, C_out=96*x, dims=3, K=16, D=2, P=128, r_indices_func=knn_indices_func_gpu)

        self.xdconv3 = Xdconv(C_in=96*x, C_out=64*x, dims=3, K=16, D=2, P=384, r_indices_func=knn_indices_func_gpu)
        self.xdconv2 = Xdconv(C_in=64*x, C_out=32*x, dims=3, K=16, D=1, P=768, r_indices_func=knn_indices_func_gpu)
        self.xdconv1 = Xdconv(C_in=32*x, C_out=16*x, dims=3, K=12, D=1, P=-1, r_indices_func=knn_indices_func_gpu)

        self.fc = nn.Sequential(Dense(16*x, 16*x),
                                Dense(16*x, 16*x, drop_rate=0.7),
                                Dense(16*x, num_classes, activation=None)
                                )

    def forward(self, points, features):
        """

        :param points: tensor(N, 2048, 3)
        :param features: tensor(N, 2048, 3)
        :return:
        """
        points1, features1 = self.pointcnn1((points, features))  # (N, 2048, 3), (N, 2048, 16*x)
        points2, features2 = self.pointcnn2((points1, features1))  # (N, 768, 3), (N, 768, 32*x)
        points3, features3 = self.pointcnn3((points2, features2))  # (N, 384, 3), (N, 384, 64*x)
        points4, features4 = self.pointcnn4((points3, features3))  # (N, 128, 3), (N, 128, 96*x)

        features3 = self.xdconv3((points3, features3, points4, features4))  # (N, 384, 64*x)
        features2 = self.xdconv2((points2, features2, points3, features3))  # (N, 768, 32*x)
        features1 = self.xdconv1((points1, features1, points2, features2))  # (N, 2048, 16*x)

        res = self.fc(features1)  # (N, 2048, num_classes)
        logit = F.log_softmax(res, dim=-1)  # (b, 2048, num_classes)

        return logit
