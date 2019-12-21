from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
from data.augment import get_augmentations_from_list
from models.pointSemantic import PointSemantic
from models.loss import Criterion
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from datetime import datetime
# import importlib
from utils import data_utils, basics_util
import math
from metric import ConfusionMatrix
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

cudnn.enabled = True

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int, default=1,
                    help='gpu id for network, -1 will use cpu')
parser.add_argument('--batch_size_train', type=int, default=8,
                    help='Batch Size during training [default: 32]')
parser.add_argument('--batch_size_val', type=int, default=6,
                    help='Batch Size during training [default: 32]')
parser.add_argument('--max_epoch', type=int, default=256,
                    help='Epoch to run [default: 100]')
parser.add_argument('--init_learning_rate',
                    type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 20)')
parser.add_argument('--summary_log_dir', default='summary_log/',
                    help='Log dir [default: log]')
parser.add_argument('--augmentation', type=str, nargs='+', default=['Jitter', 'Shift'],
                    choices=['Jitter', 'RotateSmall', 'Rotate1D', 'Rotate3D', 'Scale', 'Shift'],
                    help='Data augmentation settings to use during training')
parser.add_argument('--upright_axis', type=int, default=2,
                    help='Will learn invariance along this axis')
parser.add_argument('--resume_model', type=str, default='',
                    help='If present, restore checkpoint and resume training')
parser.add_argument("--seg", type=float, default=1.0,
                    help="Smooth term for position")
parser.add_argument("--cha", type=float, default=1.0,
                    help="Smooth term for translation, default=-7")
# parser.add_argument('--filelist', '-t', help='Path to training set ground truth (.txt)', required=True)
# parser.add_argument('--filelist_val', '-v', help='Path to validation set ground truth (.txt)', required=True)
parser.add_argument('--save_folder', '-s', help='Path to folder for saving check points and summary', required=True)
parser.add_argument('--log', help='Log to FILE in save folder; use - for stdout (default is log.txt)', metavar='FILE', default='log.txt')
parser.add_argument('--sample_num', help='downsample number before feed to net', type=int, default=4096)
parser.add_argument('--step_val', help='downsample number before feed to net', type=int, default=500)
parser.add_argument('--no_timestamp_folder', help='Dont save to timestamp folder', action='store_true')
parser.add_argument('--model', '-m', help='Model to use', required=True)
parser.add_argument('--use_normals', action='store_true')
args = parser.parse_args()

GPU_ID = args.gpu_id
NUM_EPOCH = args.max_epoch
BATCH_SIZE_TRAIN = args.batch_size_train
BATCH_SIZE_VAL = args.batch_size_val
SAMPLE_NUM = args.sample_num
STEP_VAL = args.step_val
TRAIN_AUGMENTATION = get_augmentations_from_list(args.augmentation, upright_axis=args.upright_axis)
RESUME_MODEL = args.resume_model
RAND_SEED = args.seed
SUMMARY_LOG_DIR = args.summary_log_dir
MODEL_OPTIMIZER = args.optimizer
INIT_LEARNING_RATE = args.init_learning_rate
NO_TIMESTAMP_FOLDER = args.no_timestamp_folder
SAVE_FOLDER = args.save_folder
MODEL_NAME = args.model
LOG = args.log
USE_NORMALS = args.use_normals

train_augmentations = get_augmentations_from_list(TRAIN_AUGMENTATION)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
if GPU_ID >= 0:

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")
else:
    device = torch.device("cpu")

if not NO_TIMESTAMP_FOLDER:
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(SAVE_FOLDER, '%s_%s_%d' % (MODEL_NAME, time_string, os.getpid()))
else:
    root_folder = SAVE_FOLDER
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

LOG_FOUT = open(os.path.join(root_folder, LOG), 'w')
LOG_FOUT.write(str(args) + '\n')

print('PID:', os.getpid())

print(args)

# Prepare inputs
num_classes = 8
print('{}-Preparing datasets...'.format(datetime.now()))
semantic3d_filelist = "/home/yss/sda1/yzl/Data/Semantic3D/train_data_files.txt"
# npm3d_filelist = "/home/yss/sda1/yzl/Data/NPM3D/train_data_files.txt"
semantic3d_val_list = "/home/yss/sda1/yzl/Data/Semantic3D/val_data_files.txt"
npm3d_val_list = "/home/yss/sda1/yzl/Data/NPM3D/val_data_files.txt"
is_list_of_h5_list = not data_utils.is_h5_list(semantic3d_filelist)

if is_list_of_h5_list:
    seg_list = data_utils.load_seg_list(semantic3d_filelist)
    seg_list_idx = 0
    semantic3d_filelist_train = seg_list[seg_list_idx]
    seg_list_idx = seg_list_idx + 1
else:
    filelist_train = semantic3d_filelist
semantic3d_data_train, _, semantic3d_data_num_train, semantic3d_label_train, _ = data_utils.load_seg(semantic3d_filelist_train)
semantic3d_data_val, _, semantic3d_data_num_val, semantic3d_label_val, _ = data_utils.load_seg(semantic3d_val_list, 1)
npm3d_data_val, _, npm3d_data_num_val, npm3d_label_val, _ = data_utils.load_seg(npm3d_val_list, 1)

# shuffle
semantic3d_data_train, semantic3d_label_train, semantic3d_label_train = data_utils.grouped_shuffle([semantic3d_data_train, semantic3d_data_num_train, semantic3d_label_train])

num_train = semantic3d_data_train.shape[0]
point_num = semantic3d_data_train.shape[1]
semantic3d_num_val = semantic3d_data_val.shape[0]
npm3d_num_val = npm3d_data_val.shape[0]
print('{}-{:d}/{:d}/{:d} training/semantic3d_validation/npm3d_validation samples.'.format(datetime.now(), num_train, semantic3d_num_val, npm3d_num_val))
batch_num = (num_train * NUM_EPOCH + BATCH_SIZE_TRAIN - 1) // BATCH_SIZE_TRAIN
print('{}-{:d} training batches.'.format(datetime.now(), batch_num))
semantic3d_batch_num_val = semantic3d_num_val // BATCH_SIZE_VAL
print('{}-{:d} semantic 3d validation batches per test.'.format(datetime.now(), semantic3d_batch_num_val))
npm3d_batch_num_val = npm3d_num_val // BATCH_SIZE_VAL
print('{}-{:d} npm3d validation batches per test.'.format(datetime.now(), npm3d_batch_num_val))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train():
    global num_train, semantic3d_data_train, semantic3d_data_num_train, semantic3d_label_train, seg_list_idx
    setup_seed(RAND_SEED)
    train_writer = SummaryWriter(os.path.join(root_folder, SUMMARY_LOG_DIR, 'train'))
    val_writer = SummaryWriter(os.path.join(root_folder, SUMMARY_LOG_DIR, 'val'))

    model = PointSemantic(num_classes, addition_dim=0)
    model = model.to(device)

    criterion = Criterion()  # no parameter
    # loss = Criterion_lr(sap=FLAGS.seg, srx=FLAGS.cha, learn_gamma=True)  # learnable parameter

    if MODEL_OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), INIT_LEARNING_RATE, weight_decay=1e-4)
    elif MODEL_OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), INIT_LEARNING_RATE, weight_decay=1e-4)
    else:
        optimizer = None
        exit(0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.95)

    if len(RESUME_MODEL) > 0:
        resume_path = os.path.join(root_folder, RESUME_MODEL)
        print("Resuming From ", resume_path)
        checkpoint = torch.load(resume_path)
        saved_state_dict = checkpoint['state_dict']
        start_iter = checkpoint['iter']
        model.load_state_dict(saved_state_dict)
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_iter = 0

    if GPU_ID >= 0:
        model = nn.DataParallel(model)  # as we only use one gpu, it does't change anything...

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    parameter_num = np.sum([np.prod(list(v.shape)) for v in model.parameters()])
    log_string('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    npm3d_batch_idx = start_iter
    for batch_idx_train in range(start_iter, batch_num):
        # validation
        if (batch_idx_train % STEP_VAL == 0 and (batch_idx_train != 0 or RESUME_MODEL is not None)) \
                or batch_idx_train == batch_num - 1:
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
            else:
                model_to_save = model
            torch.save({
                'iter': batch_idx_train,
                'state_dict': model_to_save.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(root_folder, 'checkpoint_iter{}.tar'.format(batch_idx_train)))
            print("Model Saved As " + 'checkpoint_iter{}.tar'.format(batch_idx_train))

            val_one_epoch(model,
                          [npm3d_data_val, npm3d_data_num_val, npm3d_label_val, npm3d_batch_num_val,
                           npm3d_num_val, 'npm3d'], val_writer, device, batch_idx_train)
            val_one_epoch(model,
                          [semantic3d_data_val, semantic3d_data_num_val, semantic3d_label_val, semantic3d_batch_num_val,
                           semantic3d_num_val, 'semantic3d'], val_writer, device, batch_idx_train)

        # semantic data prepare
        semantic3d_start_idx = (BATCH_SIZE_TRAIN * batch_idx_train) % num_train
        semantic3d_end_idx = min(semantic3d_start_idx + BATCH_SIZE_TRAIN, num_train)
        batch_size_train = semantic3d_end_idx - semantic3d_start_idx
        semantic3d_points_batch = semantic3d_data_train[semantic3d_start_idx:semantic3d_end_idx, ...]  # (B, N, C)
        # we only use location(x,y,z) information and ignore other information like rgb, intensity
        semantic3d_points_batch = semantic3d_points_batch[:, :, :3]  # (B, N, 3)
        semantic3d_points_num_batch = semantic3d_data_num_train[semantic3d_start_idx:semantic3d_end_idx, ...]  # (B)
        semantic3d_labels_batch = semantic3d_label_train[semantic3d_start_idx:semantic3d_end_idx, ...]  # (B, N)
        # npm data prepare
        npm3d_batch_idx += 1
        npm3d_start_idx = (BATCH_SIZE_TRAIN * npm3d_batch_idx) % npm3d_num_val
        if npm3d_start_idx + batch_size_train > npm3d_num_val:
            npm3d_start_idx = 0
            npm3d_batch_idx = 0
        npm3d_end_idx = npm3d_start_idx + batch_size_train
        npm3d_points_batch = npm3d_data_val[npm3d_start_idx:npm3d_end_idx, ...]  # (B, N, C)
        npm3d_points_batch = npm3d_points_batch[:, :, :3]  # (B, N, 3)
        npm3d_points_num_batch = npm3d_data_num_val[npm3d_start_idx:npm3d_end_idx, ...]  # (B)
        # npm3d_labels_batch = npm3d_label_val[npm3d_start_idx:npm3d_end_idx, ...]  # (B, N)

        # replace training dataset if previous training dataset is run out
        if semantic3d_start_idx + batch_size_train == num_train:
            if is_list_of_h5_list:
                filelist_train_prev = seg_list[(seg_list_idx - 1) % len(seg_list)]
                filelist_train = seg_list[seg_list_idx % len(seg_list)]
                # don't update train data if len(seg_list) == 1
                if filelist_train != filelist_train_prev:
                    semantic3d_data_train, _, semantic3d_data_num_train, semantic3d_label_train, _ = data_utils.load_seg(filelist_train)
                    num_train = semantic3d_data_train.shape[0]
                seg_list_idx = seg_list_idx + 1
            semantic3d_data_train, semantic3d_data_num_train, semantic3d_label_train = \
                data_utils.grouped_shuffle([semantic3d_data_train, semantic3d_data_num_train, semantic3d_label_train])

        batch_indices = np.arange(batch_size_train, dtype=np.long).reshape(batch_size_train, 1).repeat(SAMPLE_NUM, axis=1)  # (B, sample_num)
        # sample semantic3d
        semantic3d_sample_indices = basics_util.get_indices(batch_size_train, SAMPLE_NUM, semantic3d_points_num_batch)  # (B, sample_num)
        semantic3d_points_sampled = semantic3d_points_batch[batch_indices, semantic3d_sample_indices]  # (B, sample_num, 3)
        semantic3d_labels_sampled = semantic3d_labels_batch[batch_indices, semantic3d_sample_indices]  # (B, sample_num)
        # sample npm3d
        npm3d_sample_indices = basics_util.get_indices(batch_size_train, SAMPLE_NUM, npm3d_points_num_batch)  # (B, sample_num)
        npm3d_points_sampled = npm3d_points_batch[batch_indices, npm3d_sample_indices]  # (B, sample_num, 3)
        # npm3d_labels_sampled = npm3d_labels_batch[batch_indices, npm3d_sample_indices]  # (B, sample_num)
        # augmentation
        for idx in range(0, batch_size_train):
            for a in train_augmentations:
                semantic3d_points_sampled[idx, :, :3] = a.apply(semantic3d_points_sampled[idx, :, :3])
                npm3d_points_sampled[idx, :, :3] = a.apply(npm3d_points_sampled[idx, :, :3])

        # convert to tensor
        semantic3d_points_tensor = torch.from_numpy(semantic3d_points_sampled).to(device, dtype=torch.float32)  # (B, sample_num, 3)
        semantic3d_labels_tensor = torch.from_numpy(semantic3d_labels_sampled).to(device, dtype=torch.long)  # (B, sample_num)
        npm3d_points_tensor = torch.from_numpy(npm3d_points_sampled).to(device, dtype=torch.float32)  # (B, sample_num, 3)
        # npm3d_labels_tensor = torch.from_numpy(npm3d_labels_sampled).to(device, dtype=torch.long)  # (B, sample_num)

        # add mannual additional information
        if USE_NORMALS:
            semantic3d_normals_tensor = basics_util.compute_normals(semantic3d_points_tensor)  # (B, sample_num, 3)
            npm3d_normals_tensor = basics_util.compute_normals(npm3d_points_tensor)  # (B, sample_num, 3)
            semantic3d_points_tensor = torch.cat([semantic3d_points_tensor, semantic3d_normals_tensor], dim=2)  # (B, sample_num, 6)
            npm3d_points_tensor = torch.cat([npm3d_points_tensor, npm3d_normals_tensor], dim=2)  # (B, sample_num, 6)
        # run model and then optimize
        scheduler.optimizer.zero_grad()
        semantic3d_points_prob, npm3d_reconstructed = run_model(model, semantic3d_points_tensor, npm3d_points_tensor)  # (B, sample_num, num_classes), (B, sample_num, 3)
        _, semantic3d_points_pred = torch.max(semantic3d_points_prob, dim=2)  # (B, sample_num)
        batch_loss = criterion(semantic3d_points_prob, semantic3d_labels_tensor, npm3d_reconstructed, npm3d_points_tensor[:, :, :3])
        batch_loss.backward()
        scheduler.optimizer.step()
        scheduler.step()
        log_string('iter: %d, Loss: %f' % (batch_idx_train, batch_loss))
        train_writer.add_scalar('Loss', batch_loss.cpu().item(), batch_idx_train)


def val_one_epoch(model, dataset_relevant, val_writer, device, batch_idx_train):
    data_val, data_num_val, label_val, batch_num_val, num_val, dataset_name = dataset_relevant
    if 'npm' in dataset_name:
        val_classes = 5
    else:
        val_classes = 8
    CM = ConfusionMatrix(val_classes)
    for batch_val_idx in tqdm(range(batch_num_val // 10)):
        start_idx = BATCH_SIZE_VAL * batch_val_idx
        end_idx = min(start_idx + BATCH_SIZE_VAL, num_val)
        batch_size_val = end_idx - start_idx
        points_batch = data_val[start_idx:end_idx, ...]  # (B, N, C)
        # we only use location(x,y,z) information and ignore other information like rgb, intensity
        points_batch = points_batch[:, :, :3]  # (B, N, 3)
        points_num_batch = data_num_val[start_idx:end_idx, ...]  # (B)
        labels_batch = label_val[start_idx:end_idx, ...]  # (B, N)

        if 'npm' in dataset_name:
            sample_indices = basics_util.get_indices(batch_size_val, SAMPLE_NUM, points_num_batch)  # (B, sample_num)
        else:
            sample_indices = basics_util.get_indices(batch_size_val, SAMPLE_NUM, points_num_batch)
        batch_indices = np.arange(batch_size_val, dtype=np.long).reshape(batch_size_val, 1).repeat(SAMPLE_NUM, axis=-1)  # (B, sample_num)
        points_sampled = points_batch[batch_indices, sample_indices]  # (B, sample_num, 3)
        labels_sampled = labels_batch[batch_indices, sample_indices]  # (B, sample_num)

        # convert to tensor
        points_tensor = torch.from_numpy(points_sampled).to(device, dtype=torch.float32)
        # labels_tensor = torch.from_numpy(labels_sampled).to(device, dtype=torch.long)

        # add mannual additional information
        if USE_NORMALS:
            normals_tensor = basics_util.compute_normals(points_tensor)  # (B, sample_num, 3)
            points_tensor = torch.cat([points_tensor, normals_tensor], dim=2)  # (B, sample_num, 6)
        zero_padding = torch.zeros_like(points_tensor).to(device, dtype=torch.float32)  # (B, sample_num, 6)
        points_prob, _ = run_model(model, points_tensor, zero_padding, validate=False)  # (B, sample_num, num_classes)
        _, points_pred = torch.max(points_prob, dim=2)  # (B, sample_num)
        points_pred = points_pred.cpu().numpy()  # (B, sample_num)
        if 'npm' not in dataset_name:
            new_class_labels = labels_sampled.flatten()
            new_class_pred = points_pred.flatten()
        else:
            new_class_labels, new_class_pred = convert2new(labels_sampled.flatten(), points_pred.flatten())
        CM.count_predicted(new_class_labels, new_class_pred)
    mIOU = CM.get_average_intersection_union()
    OA = CM.get_overall_accuracy()
    log_string('%s mIOU: %f' % (dataset_name, mIOU))
    log_string('%s OA: %f' % (dataset_name, OA))
    val_writer.add_scalar('%s mIOU' % dataset_name, mIOU, batch_idx_train)
    val_writer.add_scalar('%s OA' % dataset_name, OA, batch_idx_train)


def convert2new(labels, pred):
    """
    convert semantic label and npm label to new class
    :param labels: (N,) npm groundtruth label available by dataset
    :param pred: (N,) semantic prediction label availabel by model
    :return: (n,), (n,)
    """
    semantic2new = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: -1, 7: 4}
    npm2new = {0: 0, 1: 2, 2: 3, 3: 3, 4: 3, 5: 3, 6: -1, 7: 4, 8: 1}
    N = len(labels)
    for i in range(N):
        labels[i] = npm2new[labels[i]]
        pred[i] = semantic2new[pred[i]]
    indices = np.logical_and(labels != -1, pred != -1)
    labels = labels[indices]
    pred = pred[indices]
    return labels, pred


def run_model(model, P, Q, validate=False):

    if not validate:
        model = model.train()
        return model(P, Q)
    else:
        with torch.no_grad():
            model = model.eval()
            return model(P, Q)


if __name__ == "__main__":
    train()
