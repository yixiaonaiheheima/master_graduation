from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.datagenerator import DataGenerator
from data.augment import get_augmentations_from_list
from models.model import PointSemantic
from models.loss import Criterion, Criterion_lr
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

cudnn.enabled = True

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int, default=1,
                    help='gpu id for network, -1 will use cpu')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch Size during training [default: 32]')
parser.add_argument('--max_epoch', type=int, default=256,
                    help='Epoch to run [default: 100]')
parser.add_argument('--init_learning_rate',
                    type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--seed', type=int, default=999, metavar='S',
                    help='random seed (default: 20)')
parser.add_argument('--log_dir', default='./log/',
                    help='Log dir [default: log]')
parser.add_argument('--dataset_folder', default='/home/yss/sda1/yss/Data/Oxford/',
                    help='Our Dataset Folder：Oxford or NCLT')
parser.add_argument('--num_points', type=int, default=4096,
                    help='Number of points to downsample model to')
parser.add_argument('--augmentation', type=str, nargs='+', default=['Jitter', 'RotateSmall', 'Shift', 'Scale'],
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
parser.add_argument('--filelist', '-t', help='Path to training set ground truth (.txt)', required=True)
parser.add_argument('--filelist_val', '-v', help='Path to validation set ground truth (.txt)', required=True)
parser.add_argument('--save_folder', '-s', help='Path to folder for saving check points and summary', required=True)
parser.add_argument('--log', help='Log to FILE in save folder; use - for stdout (default is log.txt)', metavar='FILE', default='log.txt')
FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.log_dir):
    os.mkdir(FLAGS.log_dir)
LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

TOTAL_ITERATIONS = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
if FLAGS.gpu_id >= 0:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")
else:
    device = torch.device("cpu")

# train
train_file = os.path.join(FLAGS.dataset_folder, 'train/train.txt')
train_data = DataGenerator(train_file, num_cols=6)
train_augmentations = get_augmentations_from_list(FLAGS.augmentation, upright_axis=FLAGS.upright_axis)
# test
test_file = os.path.join(FLAGS.dataset_folder, 'test/test.txt')
test_data = DataGenerator(test_file, num_cols=6)


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
    global TOTAL_ITERATIONS
    setup_seed(FLAGS.seed)
    train_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'test'))

    model = PointSemantic(8)
    model = model.to(device)

    loss = Criterion()  # no parameter
    # loss = Criterion_lr(sap=FLAGS.seg, srx=FLAGS.cha, learn_gamma=True)  # learnable parameter

    if FLAGS.optimizer == 'momentum':
        # optimizer = torch.optim.SGD([{"params": model.parameters(), "lr": FLAGS.init_learning_rate},
        #                              {"params": loss.parameters(), "lr": FLAGS.init_learning_rate}])
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.init_learning_rate)
    elif FLAGS.optimizer == 'adam':
        # optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": FLAGS.init_learning_rate},
        #                               {"params": loss.parameters(), "lr": FLAGS.init_learning_rate}])
        optimizer = torch.optim.Adam(model.parameters(), FLAGS.init_learning_rate)
    else:
        optimizer = None
        exit(0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.95)

    if len(FLAGS.resume_model) > 0:
        resume_filename = FLAGS.log_dir + FLAGS.resume_model
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch'] + 1
        TOTAL_ITERATIONS = starting_epoch * len(train_file)

        model.load_state_dict(saved_state_dict)
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        starting_epoch = 0

    if FLAGS.gpu_id >= 0:
        model = nn.DataParallel(model)  # as we only use one gpu, it does't change anything...

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    for epoch in range(starting_epoch, FLAGS.max_epoch):
        log_string('**** EPOCH %03d ****' % epoch)
        sys.stdout.flush()
        train_one_epoch(model, train_data, test_data, scheduler, epoch, train_writer, test_writer, loss, device)


def train_one_epoch(model, train_data, test_data, scheduler, epoch, train_writer, test_writer, loss, device):
    global TOTAL_ITERATIONS
    train_data.shuffle()
    for step in range(train_data.size // FLAGS.batch_size):
        TOTAL_ITERATIONS += 1

        if step % 500 == 0:
            test_one_epoch(model, test_data, epoch, test_writer, device)

        anc, pos, anc_p, pos_p, x, q = train_data.next_batch(k=FLAGS.batch_size, num_points=FLAGS.num_points,
                                                             augmentation=train_augmentations)
        anc_tensor = torch.from_numpy(anc).to(device, dtype=torch.float32)
        pos_tensor = torch.from_numpy(pos).to(device, dtype=torch.float32)
        gt_anc_p = torch.from_numpy(anc_p).to(device, dtype=torch.float32)
        gt_pos_p = torch.from_numpy(pos_p).to(device, dtype=torch.float32)
        gt_x = torch.from_numpy(x).to(device, dtype=torch.float32)
        gt_q = torch.from_numpy(q).to(device, dtype=torch.float32)
        # 修改！！！

        scheduler.optimizer.zero_grad()
        seg_pred, point_pred = run_model(model, anc_tensor, pos_tensor, validate=False)
        train_loss = loss(seg_pred, seg, point_pred, point)
        train_loss.backward()
        scheduler.step()
        scheduler.optimizer.step()
        log_string('Loss: %f' % train_loss)
        train_writer.add_scalar('Loss', train_loss.cpu().item(), TOTAL_ITERATIONS)

    if epoch % 1 == 0:
        if isinstance(model, nn.DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
        torch.save({
            'epoch': epoch,
            'iter': TOTAL_ITERATIONS,
            'state_dict': model_to_save.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
            FLAGS.log_dir+'checkpoint_epoch{}.tar'.format(epoch))
        print("Model Saved As " + 'checkpoint_epoch{}.tar'.format(epoch))


def test_one_epoch(model, test_data, epoch, test_writer, device):
    test_data.shuffle()
    trees = test_data.trees
    error_results, regs_acc, = [], []

    if epoch >= 10:
        test_batch = 1
    else:
        test_batch = FLAGS.batch_size

    top1_acc_count = 0
    top5_acc_count = 0
    test_count = test_data.size // test_batch
    start = time.time()

    for step in range(test_count):
        val_flag = True
        anc, pos, anc_p, pos_p, x, q, anc_idx, anc_idx_in_folder = \
            test_data.next_batch(k=test_batch, num_points=FLAGS.num_points, augmentation=[], validate=val_flag,
                                 validate_size=1, ret_fst_anc_idx=True)
        anc_tensor = torch.from_numpy(anc).to(device, dtype=torch.float32)
        pos_tensor = torch.from_numpy(pos).to(device, dtype=torch.float32)
        gt_anc_p = torch.from_numpy(anc_p).to(device, dtype=torch.float32)
        gt_pos_p = torch.from_numpy(pos_p).to(device, dtype=torch.float32)
        gt_x = torch.from_numpy(x).to(device, dtype=torch.float32)
        gt_q = torch.from_numpy(q).to(device, dtype=torch.float32)
        pred_anc_p, pred_pos_p, pred_x, pred_q = run_model(model, anc_tensor, pos_tensor, validate=val_flag)

        anc_folder_index = test_data.fileIndex2folder[anc_idx]
        tree = trees[anc_folder_index]
        query = np.array(pred_pos_p.cpu())
        ind_nn = tree.query(query, k=5, return_distance=False)

        if anc_idx_in_folder == ind_nn[0][0]:
            top1_acc_count += 1
        if anc_idx_in_folder in ind_nn[0]:
            top5_acc_count += 1

        error_ap = val_translation(pred_anc_p, gt_anc_p)
        error_rt = val_translation(pred_x, gt_x)
        error_rr = val_rotation(pred_q, gt_q)
        registration_acc = np.sum(np.logical_and(error_rt <= 2, error_rr <= 5))
        error_results.append([error_ap, error_rt, error_rr])
        regs_acc.append(registration_acc)
        log_string('APE(m): %f' % error_ap)
        log_string('RTE(m): %f' % error_rt)
        log_string('RRE(degrees): %f' % error_rr)

    end = time.time()
    mean_cost_time = (end-start) / test_count
    mean_error = np.mean(error_results, axis=0)
    mean_reg_acc = np.mean(regs_acc)
    top1_rate = top1_acc_count / test_count
    top5_rate = top5_acc_count / test_count
    log_string('Mean Cost Time(s): %f' % mean_cost_time)
    log_string('Mean Absolute Position Error(m): %f' % mean_error[0])
    log_string('Mean Relative Translation Error(m): %f' % mean_error[1])
    log_string('Mean Relative Rotation Error(degrees): %f' % mean_error[2])
    log_string('Mean Registration Accuracy: %f' % mean_reg_acc)
    log_string('Top1 Recall is %f' % top1_rate)
    log_string('Top5 Recall is %f' % top5_rate)
    test_writer.add_scalar('MAPE', mean_error[0], TOTAL_ITERATIONS)
    test_writer.add_scalar('MRTE', mean_error[1], TOTAL_ITERATIONS)
    test_writer.add_scalar('MRRE', mean_error[2], TOTAL_ITERATIONS)
    test_writer.add_scalar('Reg_acc', mean_reg_acc, TOTAL_ITERATIONS)
    test_writer.add_scalar('Top1', top1_rate, TOTAL_ITERATIONS)
    test_writer.add_scalar('Top5', top5_rate, TOTAL_ITERATIONS)


def run_model(model, P, Q, validate=False):

    if not validate:
        model.train()
        return model(P, Q)
    else:
        with torch.no_grad():
            model.eval()
            return model(P, Q)


if __name__ == "__main__":
    train()
