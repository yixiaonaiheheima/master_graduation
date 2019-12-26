from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data.semantic_dataset import SemanticDataset
from data.npm_dataset import NpmDataset
import sys
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
from data.augment import get_augmentations_from_list
from models.pointnet import PointNetDenseCls
from models.loss import PointnetCriterion
from models.pointnet2 import PointNet2Seg
from pointcnn_utils.pointcnn import PointCNN_seg
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from datetime import datetime
# import importlib
from utils import data_utils, basics_util
import math
from metric import ConfusionMatrix
from tqdm import tqdm
import json
import datetime
import numpy as np
import multiprocessing as mp
import argparse
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

cudnn.enabled = True


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int, default=1,
                    help='gpu id for network, -1 will use cpu')
parser.add_argument('--batch_size_train', type=int, default=32,
                    help='Batch Size during training [default: 32]')
parser.add_argument('--batch_size_val', type=int, default=32,
                    help='Batch Size during training [default: 32]')
parser.add_argument('--max_epoch', type=int, default=500,
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
parser.add_argument('--log', help='Log to FILE in save folder; use - for stdout (default is log.txt)', metavar='FILE', default='log.txt')
parser.add_argument('--sample_num', help='downsample number before feed to net', type=int, default=4096)
parser.add_argument('--step_val', help='downsample number before feed to net', type=int, default=500)
parser.add_argument('--no_timestamp_folder', help='Dont save to timestamp folder', action='store_true')
parser.add_argument('--model', '-m', help='Model to use', required=True)
parser.add_argument('--use_normals', action='store_true')
parser.add_argument("--train_set", default="train", help="train, train_full")
# parser.add_argument("--config_file", default="semantic_no_color.json", help="config file path")
parser.add_argument("--dataset_name", default="npm", help="npm, semantic")
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
MODEL_NAME = args.model
LOG = args.log
USE_NORMALS = args.use_normals
DATASET_NAME = args.dataset_name
if DATASET_NAME == "npm":
    PARAMS = json.loads(open("npm.json").read())
    # Import dataset
    TRAIN_DATASET = NpmDataset(
        num_points_per_sample=PARAMS["num_point"],
        split=args.train_set,
        box_size_x=PARAMS["box_size_x"],
        box_size_y=PARAMS["box_size_y"],
        path=PARAMS["data_path"],
    )
    VALIDATION_DATASET = NpmDataset(
        num_points_per_sample=PARAMS["num_point"],
        split="validation",
        box_size_x=PARAMS["box_size_x"],
        box_size_y=PARAMS["box_size_y"],
        path=PARAMS["data_path"],
    )
elif DATASET_NAME == "semantic":
    PARAMS = json.loads(open("semantic_no_color.json").read())
    # Import dataset
    TRAIN_DATASET = SemanticDataset(
        num_points_per_sample=PARAMS["num_point"],
        split=args.train_set,
        box_size_x=PARAMS["box_size_x"],
        box_size_y=PARAMS["box_size_y"],
        use_color=PARAMS["use_color"],
        path=PARAMS["data_path"],
    )
    VALIDATION_DATASET = SemanticDataset(
        num_points_per_sample=PARAMS["num_point"],
        split="validation",
        box_size_x=PARAMS["box_size_x"],
        box_size_y=PARAMS["box_size_y"],
        use_color=PARAMS["use_color"],
        path=PARAMS["data_path"],
    )
else:
    raise ValueError

num_classes = TRAIN_DATASET.num_classes

train_augmentations = get_augmentations_from_list(TRAIN_AUGMENTATION)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
if GPU_ID >= 0:

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")
else:
    device = torch.device("cpu")

SAVE_FOLDER = 'train_log/' + MODEL_NAME + '_' + DATASET_NAME
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


def get_batch(split):
    np.random.seed()
    if split == "train":
        return TRAIN_DATASET.sample_batch_in_all_files(
            BATCH_SIZE_TRAIN, augment=False
        )
    else:
        return VALIDATION_DATASET.sample_batch_in_all_files(
            BATCH_SIZE_VAL, augment=False
        )


def fill_queues(
    stack_train, stack_validation, num_train_batches, num_validation_batches
):
    """
    Args:
        stack_train: mp.Queue to be filled asynchronously
        stack_validation: mp.Queue to be filled asynchronously
        num_train_batches: total number of training batches
        num_validation_batches: total number of validationation batches
    """
    pool = mp.Pool(processes=mp.cpu_count())
    launched_train = 0
    launched_validation = 0
    results_train = []  # Temp buffer before filling the stack_train
    results_validation = []  # Temp buffer before filling the stack_validation
    # Launch as much as n
    while True:
        if stack_train.qsize() + launched_train < num_train_batches:
            results_train.append(pool.apply_async(get_batch, args=("train",)))
            launched_train += 1
        elif stack_validation.qsize() + launched_validation < num_validation_batches:
            results_validation.append(pool.apply_async(get_batch, args=("validation",)))
            launched_validation += 1
        for p in results_train:
            if p.ready():
                stack_train.put(p.get())
                results_train.remove(p)
                launched_train -= 1
        for p in results_validation:
            if p.ready():
                stack_validation.put(p.get())
                results_validation.remove(p)
                launched_validation -= 1
        # Stability
        time.sleep(0.01)


def init_stacking():
    """
    Returns:
        stacker: mp.Process object
        stack_validation: mp.Queue, use stack_validation.get() to read a batch
        stack_train: mp.Queue, use stack_train.get() to read a batch
    """
    # Queues that contain several batches in advance
    num_train_batches = TRAIN_DATASET.get_num_batches(BATCH_SIZE_TRAIN)
    num_validation_batches = VALIDATION_DATASET.get_num_batches(BATCH_SIZE_VAL)
    print("we have %d batches for train and %d batches for validation in one epoch" % (num_train_batches, num_validation_batches))
    stack_train = mp.Queue(num_train_batches)
    stack_validation = mp.Queue(num_validation_batches)
    stacker = mp.Process(
        target=fill_queues,
        args=(
            stack_train,
            stack_validation,
            num_train_batches,
            num_validation_batches,
        ),
    )
    stacker.start()
    return stacker, stack_validation, stack_train


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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train():
    setup_seed(RAND_SEED)
    train_writer = SummaryWriter(os.path.join(root_folder, SUMMARY_LOG_DIR, 'train'))
    val_writer = SummaryWriter(os.path.join(root_folder, SUMMARY_LOG_DIR, 'val'))

    # set model and criterion
    if MODEL_NAME == 'pointnet':
        model = PointNetDenseCls(num_classes)
        criterion = PointnetCriterion()
    elif MODEL_NAME == 'pointnet2':
        model = PointNet2Seg(num_classes)
        criterion = PointnetCriterion()
    elif MODEL_NAME == 'pointcnn':
        model = PointCNN_seg(num_classes)
        criterion = PointnetCriterion()
    else:
        raise ValueError
    model = model.to(device)

    if MODEL_OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), INIT_LEARNING_RATE, weight_decay=1e-4)
    elif MODEL_OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), INIT_LEARNING_RATE, weight_decay=1e-4)
    else:
        optimizer = None
        exit(0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.95)

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
        # model = model.apply(weights_init)

    if GPU_ID >= 0:
        model = nn.DataParallel(model)  # as we only use one gpu, it does't change anything...

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    parameter_num = np.sum([np.prod(list(v.shape)) for v in model.parameters()])
    log_string('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    CM = ConfusionMatrix(num_classes)
    batch_num = TRAIN_DATASET.get_num_batches(BATCH_SIZE_TRAIN) * NUM_EPOCH
    num_val = VALIDATION_DATASET.get_num_batches(BATCH_SIZE_VAL)
    stacker, stack_validation, stack_train = init_stacking()
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

            # calculate train performance
            mIOU = CM.get_average_intersection_union()
            OA = CM.get_overall_accuracy()
            log_string('train mIOU: %f' % mIOU)
            log_string('train OA: %f' % OA)
            CM.__init__(num_classes)
            val_one_epoch(model, [stack_validation, num_val, DATASET_NAME], val_writer, device, batch_idx_train, criterion)

        # data prepare
        batch_data, batch_label, batch_weights = stack_train.get()

        # normalize data
        # batch_data = basics_util.normalize_data(batch_data)  # (B, sample_num, 3)
        # convert to tensor
        points_tensor = torch.from_numpy(batch_data).to(device, dtype=torch.float32)  # (B, sample_num, 3)
        batch_label_tensor = torch.from_numpy(batch_label).to(device, dtype=torch.long)  # (B, sample_num)

        # run model and then optimize
        scheduler.optimizer.zero_grad()
        model = model.train()
        points_prob = run_model(model, points_tensor.permute(0, 2, 1))  # (B, sample_num, num_classes), (B, sample_num, 3)
        _, points_pred = torch.max(points_prob, dim=2)  # (B, sample_num)
        batch_loss = criterion(points_prob, batch_label_tensor)
        batch_loss.backward()
        scheduler.optimizer.step()
        scheduler.step()
        log_string('iter: %d, Loss: %f' % (batch_idx_train, batch_loss))
        train_writer.add_scalar('Loss', batch_loss.cpu().item(), batch_idx_train)
        points_pred = points_pred.cpu().numpy()
        new_class_labels = batch_label.flatten()
        new_class_pred = points_pred.flatten()
        CM.count_predicted(new_class_labels, new_class_pred)


def val_one_epoch(model, dataset_relevant, val_writer, device, batch_idx_train, criterion):
    stack_validation, batch_num_val, dataset_name = dataset_relevant
    val_classes = num_classes
    CM = ConfusionMatrix(val_classes)
    batch_loss_count = 0
    batch_num_count = 0
    for batch_val_idx in range(batch_num_val // 10):
        batch_data, batch_label, batch_weights = stack_validation.get()
        # normalize data
        # batch_data = basics_util.normalize_data(batch_data)  # (B, sample_num, 3)
        # convert to tensor
        batch_data_tensor = torch.from_numpy(batch_data).to(device, dtype=torch.float32)  # (B, sample_num, 3)
        batch_label_tensor = torch.from_numpy(batch_label).to(device, dtype=torch.long)

        model = model.eval()
        with torch.no_grad():
            points_prob = run_model(model, batch_data_tensor.permute(0, 2, 1))  # (B, sample_num, num_classes)
            batch_loss = criterion(points_prob, batch_label_tensor)
        # print("batch_val_idx, loss", batch_val_idx, batch_loss)
        batch_loss_count += batch_loss.cpu().numpy()
        batch_num_count += 1
        _, points_pred = torch.max(points_prob, dim=2)  # (B, sample_num)
        points_pred = points_pred.cpu().numpy()  # (B, sample_num)
        new_class_labels = batch_label.flatten()
        new_class_pred = points_pred.flatten()
        CM.count_predicted(new_class_labels, new_class_pred)
    mIOU = CM.get_average_intersection_union()
    OA = CM.get_overall_accuracy()
    ave_loss = batch_loss_count / batch_num_count
    ave_loss = ave_loss
    log_string('average val loss is %f' % ave_loss)
    log_string('%s mIOU: %f' % (dataset_name, mIOU))
    log_string('%s OA: %f' % (dataset_name, OA))
    val_writer.add_scalar('%s mIOU' % dataset_name, mIOU, batch_idx_train)
    val_writer.add_scalar('%s OA' % dataset_name, OA, batch_idx_train)


def run_model(model, P):
    if MODEL_NAME == 'pointnet':
        res, _, _ = model(P)
    elif MODEL_NAME == 'pointnet2':
        res, _ = model(P)
    elif MODEL_NAME == 'pointcnn':
        P_permute = P.permute(0, 2, 1)
        res = model(P_permute, P_permute)
    else:
        raise ValueError
    return res


if __name__ == "__main__":
    train()
