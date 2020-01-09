import open3d
import os
import glob
import numpy as np
from sklearn.neighbors import KDTree
from utils.point_cloud_util import _label_to_colors
from utils.eval_utils import _2common
from utils.point_cloud_util import load_labels
from utils.metric import ConfusionMatrix


model_name = 'pointsemantic'
from_dataset = 'npm'
to_dataset = 'npm'

root_folder = '/home/yss/sda1/yzl/yzl_graduation/result'
sub_folder = model_name + '_' + from_dataset + '2' + to_dataset
sparse_folder = os.path.join(root_folder, 'sparse', sub_folder)
raw_folder = '/home/yss/sda1/yzl/Data/' + to_dataset + '_raw'
dense_folder = os.path.join(root_folder, 'dense', sub_folder)
# raw_folder = '/home/yss/sda1/yzl/Data/' + to_dataset + '_downsampled'
# dense_folder = os.path.join(root_folder, 'downsampled', sub_folder)
os.makedirs(dense_folder, exist_ok=True)
for prob_path in glob.glob(os.path.join(sparse_folder, '*.prob')):
    base_name = os.path.basename(prob_path)
    fname_without_ext = os.path.splitext(base_name)[0]
    dense_common_pcd_path = os.path.join(dense_folder, fname_without_ext + '_common.pcd')
    if os.path.exists(dense_common_pcd_path):
        print("%s aleady exists, skipping..." % (fname_without_ext + '_common.pcd'))
        continue
    sparse_common_pcd_path = os.path.join(sparse_folder, fname_without_ext + '_common.pcd')
    sparse_common_pcd = open3d.io.read_point_cloud(sparse_common_pcd_path)
    points = np.asarray(sparse_common_pcd.points)
    del sparse_common_pcd
    raw_pcd_path = os.path.join(raw_folder, fname_without_ext + '.pcd')
    raw_pcd = open3d.io.read_point_cloud(raw_pcd_path)
    raw_points = np.asarray(raw_pcd.points)  # (N1, 3)
    del raw_pcd
    N1, N2 = len(raw_points), len(points)
    # load prob file
    print('load prob file %s' % prob_path)
    prob = np.loadtxt(prob_path, dtype=np.float32)  # (N2, num_classes)
    num_classes = prob.shape[-1]
    print('finish load %s' % prob_path)
    # start interpolation
    tree = KDTree(points)
    del points

    # calculate interpolated_prob by batch for save memory
    interpolated_label = np.zeros(N1, dtype=np.int64)
    batch_size = 1024
    for batch_idx in range(int(np.ceil(N1 / batch_size))):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N1)
        nn_dist, nn_idx = tree.query(raw_points[start_idx:end_idx, :], k=3)  # (batch_size, 3), (batch_size, 3)
        nn_recip = 1.0 / (nn_dist + 1e-8)  # (batch_size, 3)
        norm = np.sum(nn_recip, axis=1, keepdims=True)  # (batch_size, 1)
        weights = nn_recip / norm  # (batch_size, 3)
        weights_expand = np.expand_dims(weights, -1)  # (batch_size, 3, 1)
        nn_prob = prob[nn_idx, :]  # (batch_size, 3, num_classes)
        interpolated_prob = np.sum(nn_prob * weights_expand, axis=1)  # (batch_size, num_classes)
        interpolated_label[start_idx:end_idx] = np.argmax(interpolated_prob, axis=-1)  # (batch_size,)

    interpolated_common_label = _2common(interpolated_label, from_dataset)  # (N1,)
    interpolated_color = _label_to_colors(interpolated_label)  # (N1, 3)
    interpolated_common_color = _label_to_colors(interpolated_common_label)  # (N1, 3)
    del interpolated_common_label

    # evaluate interpolation result if ground truth is available
    raw_label_path = os.path.join(raw_folder, fname_without_ext + '.labels')
    if from_dataset == to_dataset and os.path.exists(raw_label_path):
        label_gt = load_labels(raw_label_path)  # (N1,)
        ConfusionMatrix = ConfusionMatrix(num_classes)
        ConfusionMatrix.increment_from_list(label_gt.flatten(), interpolated_label.flatten())
        ConfusionMatrix.print_metrics()

    label_path = os.path.join(dense_folder, fname_without_ext + '.labels')
    np.savetxt(label_path, interpolated_label, fmt="%d")
    del interpolated_label
    print('writing labels for %s' % fname_without_ext)

    # output pcd with from_dataset label
    dense_pcd = open3d.geometry.PointCloud()
    dense_pcd.points = open3d.utility.Vector3dVector(raw_points)
    dense_pcd.colors = open3d.utility.Vector3dVector(interpolated_color)
    del interpolated_color
    dense_pcd_path = os.path.join(dense_folder, fname_without_ext + '_' + from_dataset + '.pcd')
    open3d.io.write_point_cloud(dense_pcd_path, dense_pcd)
    del dense_pcd
    print('writing to %s' % dense_pcd_path)

    # output pcd with common label
    dense_common_pcd = open3d.geometry.PointCloud()
    dense_common_pcd.points = open3d.utility.Vector3dVector(raw_points)
    dense_common_pcd.colors = open3d.utility.Vector3dVector(interpolated_common_color)
    del interpolated_common_color
    # dense_common_pcd_path = os.path.join(dense_folder, fname_without_ext + '_common.pcd')
    open3d.io.write_point_cloud(dense_common_pcd_path, dense_common_pcd)
    del dense_common_pcd, raw_points
    print('writing to %s' % dense_common_pcd_path)


print('done')
