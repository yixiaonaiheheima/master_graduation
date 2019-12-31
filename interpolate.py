import open3d
import os
import glob
import numpy as np
from sklearn.neighbors import KDTree
from utils.point_cloud_util import _label_to_colors
from utils.eval_utils import _2common


model_name = 'pointnet2'
from_dataset = 'semantic'
to_dataset = 'npm'

root_folder = '/home/yss/sda1/yzl/yzl_graduation/result'
sub_folder = model_name + '_' + from_dataset + '2' + to_dataset
sparse_folder = os.path.join(root_folder, 'sparse', sub_folder)
# raw_folder = '/home/yss/sda1/yzl/Data/' + to_dataset + '_raw'
# dense_folder = os.path.join(root_folder, 'dense', sub_folder)
raw_folder = '/home/yss/sda1/yzl/Data/' + to_dataset + '_downsampled'
dense_folder = os.path.join(root_folder, 'downsampled', sub_folder)
os.makedirs(dense_folder, exist_ok=True)
for prob_path in glob.glob(os.path.join(sparse_folder, '*.prob')):
    base_name = os.path.basename(prob_path)
    fname_without_ext = os.path.splitext(base_name)[0]
    sparse_common_pcd_path = os.path.join(sparse_folder, fname_without_ext + '_common.pcd')
    sparse_common_pcd = open3d.io.read_point_cloud(sparse_common_pcd_path)
    points = np.asarray(sparse_common_pcd.points)
    raw_pcd_path = os.path.join(raw_folder, fname_without_ext + '.pcd')
    raw_pcd = open3d.io.read_point_cloud(raw_pcd_path)
    raw_points = np.asarray(raw_pcd.points)
    N1, N2 = len(raw_points), len(points)
    print('load prob file %s' % prob_path)
    prob = np.loadtxt(prob_path, dtype=np.float32)  # (N2, num_classes)
    print('finish load %s' % prob_path)
    # start interpolation
    tree = KDTree(points)
    nn_dist, nn_idx = tree.query(raw_points, k=3)  # (N1, 3), (N1, 3)
    nn_recip = 1.0 / (nn_dist + 1e-8)  # (N1, 3)
    norm = np.sum(nn_recip, axis=1, keepdims=True)  # (N1, 1)
    weights = nn_recip / norm  # (N1, 3)
    weights_expand = np.expand_dims(weights, -1)  # (N1, 3, 1)
    nn_prob = prob[nn_idx, :]  # (N1, 3, num_classes)
    interpolated_prob = np.sum(nn_prob * weights_expand, axis=1)  # (N1, num_classes)
    interpolated_label = np.argmax(interpolated_prob, axis=-1)  # (N1,)
    interpolated_common_label = _2common(interpolated_label, from_dataset)
    interpolated_color = _label_to_colors(interpolated_label)  # (N1, 3)
    interpolated_common_color = _label_to_colors(interpolated_common_label)  # (N1, 3)

    # output pcd with from_dataset label
    dense_pcd = open3d.geometry.PointCloud()
    dense_pcd.points = open3d.utility.Vector3dVector(raw_points)
    dense_pcd.colors = open3d.utility.Vector3dVector(interpolated_color)
    dense_pcd_path = os.path.join(dense_folder, fname_without_ext + '_' + from_dataset + '.pcd')
    open3d.io.write_point_cloud(dense_pcd_path, dense_pcd)
    print('writing to %s' % dense_pcd_path)

    # output pcd with common label
    dense_common_pcd = open3d.geometry.PointCloud()
    dense_common_pcd.points = open3d.utility.Vector3dVector(raw_points)
    dense_common_pcd.colors = open3d.utility.Vector3dVector(interpolated_common_color)
    dense_common_pcd_path = os.path.join(dense_folder, fname_without_ext + '_common.pcd')
    open3d.io.write_point_cloud(dense_common_pcd_path, dense_common_pcd)
    print('writing to %s' % dense_common_pcd_path)


print('done')
