import logging
import numpy as np
import random
import os
from collections import deque
from utils import basics_util
from sklearn.neighbors import KDTree

class DataGenerator(object):

    def __init__(self, filename, num_cols=6):
        """
        Constructor to data generator
        Args:
            num_cols (int): Number of columns in binary file
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_folder = os.path.split(filename)[0]
        self.meta_datas = []
        self.inner_indices = []
        self.inverse_inner_indices = []
        self.sub_folder_index = dict()
        self.trees = []
        self.load_meta_data(self.dataset_folder)
        self.logger.info('Loaded metadata file')
        self.paths_and_labels = []
        self.icp_compose = []
        self.fileIndex2folder = []  # file index is progressively increased, so list is enough
        self.load_train(filename)
        icp_fname = os.path.join(self.dataset_folder, 'icp_compose.txt')
        self.load_icp(icp_fname)

        self.num_cols = num_cols
        self.size = len(self.paths_and_labels)  # Number of data instances
        self.indices = deque(range(self.size))

        # self.data = [None] * self.size  # not used

    def load_train(self, path):
        with open(path) as f:
            for line in f:
                fname, positives, nonegatives = [l.strip() for l in line.split('|')]
                positives = [int(s) for s in positives.split()]
                self.paths_and_labels.append((fname, positives))
                # prepare for test
                folder = fname.split('/')[0]
                self.fileIndex2folder.append(self.sub_folder_index[folder])

    def load_icp(self, path):
        with open(path) as f:
            for line in f:
                icp_trans = [l.strip() for l in line.split('|')]
                icp_trans = icp_trans[1:-1]  # first element is fname, last element is None
                icp_trans = [np.array([float(s) for s in trans.split('\t')]) for trans in icp_trans]
                self.icp_compose.append(icp_trans)

    def load_meta_data(self, dataset_folder):
        sub_folders = os.listdir(dataset_folder)
        sub_folder_count = 0
        for sub_folder in sub_folders:
            if os.path.isfile(os.path.join(dataset_folder, sub_folder)):
                continue
            self.sub_folder_index[sub_folder] = sub_folder_count
            sub_folder_count += 1
            inner_idx = []
            inverse_innver_idx = dict()
            inner_count = 0
            with open(os.path.join(dataset_folder, sub_folder, 'metadata.txt')) as f:
                f.readline()
                positions = []
                for line in f:
                    Idx, dataset, startIdx, EndIdx, NumPts, X, Y, Z = [l.strip() for l in line.split('\t')]
                    positions.append([float(X), float(Y), float(Z)])
                    Idx = int(Idx)
                    inner_idx.append(Idx)
                    inverse_innver_idx[Idx] = inner_count
                    inner_count += 1
                self.meta_datas.append(positions)
                self.inner_indices.append(inner_idx)
                self.inverse_inner_indices.append(inverse_innver_idx)
                # create kd-tree
                tree = KDTree(positions)
                self.trees.append(tree)

    def reset(self):
        """
        Resets the data generator, so that it returns the first instance again.
        Either this or shuffle() should be called
        """
        self.indices = deque(range(self.size))

    def shuffle(self):
        ''' Shuffle training data. This function should be called at the start of each epoch
        '''
        ind = list(range(self.size))
        random.shuffle(ind)
        self.indices = deque(ind)

    def next_batch(self, k=1, num_points=4096, augmentation=[], validate=False, validate_size=1, ret_fst_anc_idx=False):
        """
        Retrieves the next triplet(s) for training
        Args:
            k (int): Number of triplets
            num_points: Number of points to downsample pointcloud to
            augmentation: Types of augmentation to perform
            validate: validate flag
            validate_size: batch_size in validation(less than k), only used when validate is True(now must be 1)
            ret_fst_anc_idx: whethre return first anchor index
        Returns:
            anchors: (k, num_points, 3)
            positives: (k, num_points, 3)
            anchors_pos: (k, 3)
            positives_pos:: (k, 3)
            xs: (k, 3)
            qs: (k, 4)
            first_anchor_index: first anchor index in all files
            first_anchor_index_in_folder: first anchor index in its folder
        """

        anchors, positives, anchors_pos, positives_pos, xs, qs = [], [], [], [], [], []
        first_anchor_index = None
        first_anchor_index_in_folder = None
        for _ in range(k):
            try:
                i_anchor = self.indices.popleft()
                if first_anchor_index is None:
                    first_anchor_index = i_anchor
                i_positive, i_positive_idx = self.get_positive(i_anchor)
            except IndexError:
                break

            anchor, anchor_pos, anc_file_idx_in_folder = self.get_point_cloud(i_anchor, ret_inner_idx=True)  # (N_anc, num_cols) (3, )
            if first_anchor_index_in_folder is None:
                first_anchor_index_in_folder = anc_file_idx_in_folder
            positive, positive_pos = self.get_point_cloud(i_positive)  # (N_anc, num_cols) (3, )
            positive_icp_flat = self.icp_compose[i_anchor][i_positive_idx]
            positive_icp_flat = positive_icp_flat.reshape(1, -1)
            icp = basics_util.RT2QT(positive_icp_flat)  # (1, 7)
            anchor = self.process_point_cloud(anchor, num_points=num_points, validate=validate)
            positive = self.process_point_cloud(positive, num_points=num_points, validate=validate)

            # no rand
            # x = icp[:, :3]
            # q = icp[:, 3:]
            # x = np.squeeze(x)  # (3,)
            # q = np.squeeze(q)  # (4,)

            # rand
            x = icp[:, :3]
            q0 = icp[:, 3:]
            x = np.squeeze(x)  # (3,)
            q0 = np.squeeze(q0)  # (4,)
            anchor_cloud, anchor_norm, anchor_r = RotateZ(anchor[:, :3], anchor[:, 3:])
            anchor[:, :3] = anchor_cloud
            anchor[:, 3:] = anchor_norm
            qa = basics_util.R2Q(anchor_r)
            qa = np.squeeze(qa)
            q = basics_util.qmult(qa, q0)

            for a in augmentation:
                anchor[:, :3] = a.apply(anchor[:, :3])
                positive[:, :3] = a.apply(positive[:, :3])

            anchors.append(anchor)
            positives.append(positive)
            anchors_pos.append(anchor_pos)
            positives_pos.append(positive_pos)
            xs.append(x)
            qs.append(q)

        if len(anchors) != 0:
            anchors = np.stack(anchors, axis=0)  # (k, num_points, 3)
            positives = np.stack(positives, axis=0)  # (k, num_positives, 3)
            anchors_pos = np.stack(anchors_pos, axis=0)  # (k, 3)
            positives_pos = np.stack(positives_pos, axis=0)  # (k, 3)
            xs = np.stack(xs, axis=0)  # (k, 3)
            qs = np.stack(qs, axis=0)  # (k, 4)

            if validate:
                anchors = anchors[:validate_size, :, :]
                positives = positives[:validate_size, :, :]
                anchors_pos = anchors_pos[:validate_size, :]
                positives_pos = positives_pos[:validate_size, :]
                xs = xs[:validate_size, :]
                qs = qs[:validate_size, :]
        else:
            anchors, positives, anchors_pos, positives_pos, xs, qs = None, None, None, None, None, None

        if ret_fst_anc_idx:
            return anchors, positives, anchors_pos, positives_pos, xs, qs, first_anchor_index, first_anchor_index_in_folder
        else:
            return anchors, positives, anchors_pos, positives_pos, xs, qs

    def get_point_cloud(self, i, ret_inner_idx=False):
        """
        Retrieves the i'th point cloud
        Args:
            i (int): Index of point cloud to retrieve
            ret_inner_idx: whether return file inner index
        Returns:
            cloud (np.array) point cloud containing N points, each of D dim
            position(np.array) point cloud center position, (3, )
            file_idx: file index in current folder
        """
        assert(0 <= i < self.size)

        cloud = DataGenerator.load_point_cloud(os.path.join(self.dataset_folder, self.paths_and_labels[i][0]),
                                               num_cols=self.num_cols)
        sub_folder, file_name = self.paths_and_labels[i][0].split('/')
        file_idx = int(file_name.split('.')[0])
        folder_index = self.sub_folder_index[sub_folder]
        inner_index = self.inverse_inner_indices[folder_index][file_idx]
        position = np.array(self.meta_datas[folder_index][inner_index])
        if ret_inner_idx:
            return cloud, position, inner_index
        else:
            return cloud, position

    def get_positive(self, anchor):
        """
        Gets positive and negative indices
        Args:
            anchor (int): Index of anchor point cloud

        Returns:
            positive (int), negative (int)
        """

        _, positives = self.paths_and_labels[anchor]
        pos_indices = list(range(len(positives)))
        np.random.shuffle(pos_indices)
        idx = pos_indices[0]
        positive = positives[idx]

        # original version
        # positive = random.sample(positives, 1)[0]

        return positive, idx

    def process_point_cloud(self, cloud, num_points=4096, validate=False):
        """
        Crop and randomly downsamples of point cloud.
        """

        # Crop to 20m radius (30m for validate)
        radius = 20
        target_num = num_points
        if validate:
            radius = 20
            target_num = 4096
        mask = np.sum(np.square(cloud[:, :3]), axis=1) <= radius * radius
        cloud = cloud[mask, :]

        # Downsample
        if cloud.shape[0] <= target_num:
            # Add in artificial points if necessary
            self.logger.warning('Only %i out of %i required points in raw point cloud. Duplicating...', cloud.shape[0],
                                target_num)

            num_to_pad = target_num - cloud.shape[0]
            pad_points = cloud[np.random.choice(cloud.shape[0], size=num_to_pad, replace=True), :]
            cloud = np.concatenate((cloud, pad_points), axis=0)

            return cloud
        else:
            cloud = cloud[np.random.choice(cloud.shape[0], size=target_num, replace=False), :]
            return cloud

    @staticmethod
    def load_point_cloud(path, num_cols=6):
        """
        Reads point cloud, in our binary/text format
        Args:
            path (str): Path to .bin or .txt file
                        (bin will be assumed to be binary, txt will be assumed to be in ascii comma-delimited)
            num_cols: Number of columns. This needs to be specified for binary files.

        Returns:
            np.array of size Nx(num_cols) containing the point cloud.
        """
        if path.endswith('bin'):
            model = np.fromfile(path, dtype=np.float32)
            model = np.reshape(model, (-1, num_cols))

        else:
            model = np.loadtxt(path, dtype=np.float32, delimiter=',')

        return model


def RotateZ(cloud, norm):
    '''
    Rotation perturbation around Z-axis.
    '''
    rotation_angle = np.random.uniform(-4/16 * np.pi, 4/16 * np.pi)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])

    rotated_cloud = np.dot(cloud, np.transpose(rotation_matrix))
    rotated_norm = np.dot(norm, np.transpose(rotation_matrix))

    return rotated_cloud, rotated_norm, rotation_matrix


if __name__ == '__main__':
    train_data = DataGenerator('/home/yss/sda1/yss/Data/Oxford/train/', num_cols=6)