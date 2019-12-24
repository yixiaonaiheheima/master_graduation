import torch
from torch import nn
import torch.nn.functional as F
from utils.basics_util import square_distance


# def distChamfer(a, b):
#     """
#     Input:
#         a: a tensor (B, N1, C)
#         b: a tensor (B, N2, C)
#     return:
#          (B, N1), (B, N2)
#     """
#     x, y = a, b
#     bs, num_points, points_dim = x.size()
#     _, N2, _ = y.size()
#     xx = torch.bmm(x, x.transpose(2, 1))  # (B, N1, N1)
#     yy = torch.bmm(y, y.transpose(2, 1))  # (B, N2, N2)
#     zz = torch.bmm(x, y.transpose(2, 1))  # (B, N1, N2)
#     diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)  # （N1,)
#     diag_ind2 = torch.arange(0, N2).type(torch.cuda.LongTensor)  # （N2,)
#     rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)  # (B, N1, N1)
#     ry = yy[:, diag_ind2, diag_ind2].unsqueeze(1).expand_as(yy)  # (B, N2, N2)
#     P = (rx.transpose(2, 1) + ry - 2 * zz)
#     return P.min(1)[0], P.min(2)[0]  # (B, N) (B, N)


def distChamfer(x, y):
    """
    Input:
        x: a tensor (B, N1, C)
        y: a tensor (B, N2, C)
    return:
         (B, N1), (B, N2)
    """
    pair_dist = square_distance(x, y)  # (B, N1, N2)
    dist_x = pair_dist.min(2)[0]  # (B, N1)
    dist_y = pair_dist.min(1)[0]  # (B, N2)
    return dist_x, dist_y


class PointnetCriterion(nn.Module):
    def __init__(self):
        super(PointnetCriterion, self).__init__()

    def forward(self, seg_prob, seg_label):
        """
        calculate semantic segmentation and chamfer distance
        :arg
        seg_prob: tensor(B, N, num_class),  log-probabilities of each class
        seg_label: Long tensor(B, N)
        :return scalar
        """
        # semantic segmentation loss
        if len(seg_prob.shape) == 3:
            B, N, num_classes = seg_prob.shape
            seg_prob = seg_prob.view(B*N, num_classes)
            seg_label = seg_label.view(B*N)
        seg_loss = F.nll_loss(seg_prob, seg_label, reduction='none')  # (B,)
        seg_loss = torch.mean(seg_loss)  # scalar

        return seg_loss


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.seg = 1
        self.cha = 1

    def forward(self, seg_prob, seg_label, point_rec, point):
        """
        calculate semantic segmentation and chamfer distance
        :arg
        seg_prob: tensor(B, N, num_class),  log-probabilities of each class
        seg_label: Long tensor(B, N)
        point_rec: tensor(B, N, 3)
        point: tensor(B, N, 3), original point cloud
        :return (B
        """
        # semantic segmentation loss
        seg_loss = F.nll_loss(seg_prob.permute(0, 2, 1), seg_label, reduction='none')  # (B, N)
        seg_loss = torch.mean(seg_loss)  # scalar

        # chamfer distance loss
        dist1, dist2 = distChamfer(point_rec, point)  # (b, N), (b, N)
        cha_loss = torch.mean(dist1) + torch.mean(dist2)  # scalar

        # all loss
        loss = self.seg * seg_loss + self.cha * cha_loss

        return loss


class Criterion_lr(nn.Module):
    def __init__(self, seg=1.0, cha=1.0, learn_gamma=True):
        super(Criterion_lr, self).__init__()
        self.seg = nn.Parameter(torch.tensor([seg], requires_grad=learn_gamma, device='cuda:0'))

        self.cha = nn.Parameter(torch.tensor([cha], requires_grad=learn_gamma, device='cuda:0'))

    def forward(self, seg_pred, seg, point_pred, point):
        """
        learnable parameter
        calculate semantic segmentation and chamfer distance
        """
        # semantic segmentation loss
        seg_loss = torch.exp(-self.seg) * F.nll_loss(seg_pred, seg) + self.seg

        # chamfer distance loss
        cha_loss = torch.exp(-self.cha) * distChamfer(point_pred, point) + self.cha

        # total loss
        loss = seg_loss * 1 + cha_loss * 1

        # loss.backward()
        # if self.sap.grad is not None:
        #     print('sap: %2f' % self.sap)

        return loss
