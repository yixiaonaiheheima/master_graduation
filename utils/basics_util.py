import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import transforms3d.quaternions as txq
import random
import time


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if m > 0.1:
            pc = pc / m
        normal_data[b] = pc
    return normal_data


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    return dist


def feature_distance(P, Q):
    """
    euclidian dist between P and Q
    Input:
        P:  a float tensor (B, N, C)
        Q: a float tensor (B, N, C)
    return:
        dist: (B, N, N)/(B, N)
    """
    # P = torch.unsqueeze(P, 2)
    # Q = torch.unsqueeze(Q, 1)
    # dist = torch.sub(P, Q)
    # dist = torch.pow(dist, 2)
    # dist = torch.sum(dist, 3)  # [B, N, N]
    dist = 2 * (1 - torch.sum(P * Q, -1))  # (B, N)
    # dist = 2 - 2 * (descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)).squeeze() # d2net
    # dist = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))  # cvmnet

    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, np] or [B, np, ns]
    Return:
        new_points:, indexed points data, [B, np, C] or [B, np, ns, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # (B, 1) or (B, 1, 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # (1, np) or (1, np, ns)
    # make batch_indeces have same dimensions as view_shape
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # [B, np] or [B, np, ns]
    new_points = points[batch_indices, idx, ...]

    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # [B, ]
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # [B, ]
    for i in range(npoint):
        centroids[:, i] = farthest  # [B, N]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  # [B, ]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # [B, S, N]
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [B, S, nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])  # [B, S, nsample]
    mask = group_idx == N
    # temp1 = torch.sum(mask, -1) > 0
    # temp2 = torch.sum(temp1).to(torch.float32)
    # padding_rate = temp2 / (B * S)
    # print("padding rate is:", padding_rate.item())
    # if the point number in the sphere less than nsample, pad with group_first
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, normalize_radius=False, returnfps=False):
    """
    Input:
        npoint: keyponts number to sample
        radius: sphere radius in a group
        nsample: how many points to group for a sphere
        xyz: input points position data, [B, N, C]
        points: additional input points data, [B, N, D]
        normalize_radius: scale normalization
        returnfps: whether return FPS result
    Return:
        new_xyz: sampled points position data, [B, npoint, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, npoint, nsample, C] translation normalization

    if normalize_radius:
        grouped_xyz_norm /= radius

    if points is not None:
        grouped_points = index_points(points, idx)  # [B, np, ns, D]
        fps_points = index_points(points, fps_idx)  # [B,np, D]
        fps_points = torch.cat([new_xyz, fps_points], dim=-1)  # [B, np, C+D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm  # [B, np, ns, C]
        fps_points = new_xyz  # [B, np, C]

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points, returnfps=False):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    _, _, D = points.shape
    fps_idx = farthest_point_sample(xyz, 1)  # [B, 1]
    new_xyz = index_points(xyz, fps_idx)  # [B, 1, C]
    grouped_xyz = xyz.view(B, 1, N, C)

    if points is not None:
        grouped_points = points.view(B, 1, N, D)  # [B, 1, N, D]
        fps_points = index_points(points, fps_idx)  # [B, 1, D]
        fps_points = torch.cat([new_xyz, fps_points], dim=-1)  # [B, 1, C+D]
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points
    else:
        return new_xyz, new_points


def make_box():
    """
    function to make grids on a 3D unit box
    @param lower: lower bound
    @param upper: upper bound
    @param num: number of points on an axis. Default 18
    rvalue: 2D numpy array of dim0 = num**2*6, num1 = 3. Meaning a point cloud
    """
    lower = -0.5
    upper = 0.5
    num = 18
    a = np.linspace(lower, upper, num)
    b = np.linspace(lower, upper, num)
    grid = np.transpose([np.tile(a, len(b)), np.repeat(b, len(a))])

    c1 = np.repeat(0.5, len(grid))
    c1 = np.reshape(c1, (len(c1), -1))
    c2 = np.repeat(-0.5, len(grid))
    c2 = np.reshape(c2, (len(c2), -1))

    up = np.hstack((grid, c1))  # upper face, z == 0.5
    low = np.hstack((grid, c2))  # lower face, z == -0.5
    front = up[:, [0, 2, 1]]  # front face, y == 0.5
    back = low[:, [0, 2, 1]]  # back face, y == -0.5
    right = up[:, [2, 0, 1]]  # right face, x == 0.5
    left = low[:, [2, 0, 1]]  # left face, x == -0.5

    six_faces = np.vstack((front, back, right, left, up, low))
    return six_faces


def make_cylinder():
    """
    function to make a grid from a cyliner centered at (0, 0, 0). The cyliner's radius is 1, height is 0.5
    Method:
    1) the surrounding surface is 4 times the area of the upper and lower cicle. So we sample 4 times more points from it
    2) to match with the box, total number of points is 1944
    3) for the upper and lower surface, points are sampled with fixed degree and fixed distance along the radius
    4) for the middle surface, points are sampled along fixed lines along the height
    """
    # make the upper and lower face, which is not inclusive of the boundary points
    theta = 10  # dimension
    n = 9  # number of points for every radius
    r = 0.5
    radius_all = np.linspace(0, 0.5, n + 2)[1:10]  # radius of sub-circles
    res = []
    for i, theta in enumerate(range(0, 360, 10)):
        x = math.sin(theta)
        y = math.cos(theta)
        for r in radius_all:
            res.append([r * x, r * y])
    # add z axis
    z = np.reshape(np.repeat(0.5, len(res)), (len(res), -1))
    upper = np.hstack((np.array(res), z))  # upper face
    z = np.reshape(np.repeat(-0.5, len(res)), (len(res), -1))
    lower = np.hstack((np.array(res), z))  # lower face

    # design of middle layer: theta = 5 degree, with every divide is 18 points including boundaries
    height = np.linspace(-0.5, 0.5, 18)
    res = []
    for theta in range(0, 360, 5):
        x = 0.5 * math.sin(theta)
        y = 0.5 * math.cos(theta)
        for z in height:
            res.append([x, y, z])
    middle = np.array(res)

    cylinder = np.vstack((upper, lower, middle))
    return cylinder


def make_sphere():
    """
    function to sample a grid from a sphere
    """
    theta = np.linspace(0, 360, 36)  # determining x and y
    phi = np.linspace(0, 360, 54)  # determining z

    res = []
    for p in phi:
        z = math.sin(p) * 0.5
        r0 = math.cos(p) * 0.5
        for t in theta:
            x = math.sin(t) * r0
            y = math.cos(t) * r0
            res.append([x, y, z])

    sphere = np.array(res)
    return sphere


def get_indices(batch_size, sample_num, point_num, filter_label=None, labels=None):
    if not isinstance(point_num, np.ndarray):
        point_nums = np.full(batch_size, point_num)
    else:
        point_nums = point_num

    indices = []
    for i in range(batch_size):
        pt_num = point_nums[i]
        if filter_label is None or labels is None:
            pool_indices = np.arange(pt_num)
        else:
            label = labels[i]
            pool_indices = np.arange(pt_num)
            pool_indices = pool_indices[label[:pt_num] != filter_label]
        if len(pool_indices) >= sample_num:
            choices = np.random.choice(pool_indices, sample_num, replace=False)
        else:
            choices = np.concatenate((np.random.choice(pool_indices, len(pool_indices), replace=False),
                                      np.random.choice(pool_indices, sample_num - len(pool_indices), replace=True)))
        indices.append(choices)
    return np.stack(indices)  # (B, sample_num)


def get_cov(points):
    """
    get covariance from points
    :param points: tensor (B, N, num_neighbors,3)
    :return: (B, N, 3, 3)
    """
    points -= torch.mean(points, dim=-1, keepdim=True)
    points_t = points.permute(0, 1, 3, 2)  # (B, N, 3, num_neighbors)
    return torch.matmul(points_t, points)  # (B, N, 3, 3)


def compute_normals(cloud):
    """
    compute normals with pytorch
    :param cloud: tensor (B, N, 3)
    :return: tensor (B, N, 3)
    """
    cloud = cloud[:, :, :3]
    B, N, _ = cloud.shape
    dist = square_distance(cloud, cloud)  # (B, N, N)
    _, neighbor_indices = torch.topk(dist, 30, dim=2, largest=False, sorted=False)  # (B, N, 30)
    neighborhood = index_points(cloud, neighbor_indices)  # (B, N, 30, 3)
    cov = get_cov(neighborhood)  # (B, N, 3, 3)
    eigen_values, eigen_vectors = torch.symeig(cov, eigenvectors=True)  # (B, N, 3), (B, N, 3, 3)
    eigen_values = eigen_values.to(cloud.device)
    # we only use the real part and ignore imaginary part
    _, mini_eigen_indices = torch.min(eigen_values, dim=-1)  # (B, N)
    batch_indices = torch.arange(B, dtype=torch.long).view(B, 1).repeat(1, N)  # (B, N)
    point_indices = torch.arange(N, dtype=torch.long).view(1, N).repeat(B, 1)  # (B, N)
    normals = eigen_vectors[batch_indices, point_indices, :, mini_eigen_indices]  # (B, N, 3)

    return normals
