from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data.semantic_dataset import SemanticDataset
from data.npm_dataset import NpmDataset
import sys
import os
import numpy as np
import torch
from data.augment import get_augmentations_from_list
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import json
import datetime
import multiprocessing as mp
import argparse
import time
from datetime import datetime
from utils import metric
from utils.model_util import run_model, select_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

cudnn.enabled = True


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int, default=0, help='gpu id for network')
parser.add_argument('--batch_size_train', type=int, default=6,
                    help='Batch Size during training [default: 32]')
parser.add_argument('--batch_size_val', type=int, default=16,
                    help='Batch Size during training [default: 32]')
parser.add_argument('--max_epoch', type=int, default=500,
                    help='Epoch to run [default: 100]')
parser.add_argument('--init_learning_rate',
                    type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--lr_decay_step', type=int, default=200000,
                    help='learning rate step for every decay [default: 200000]')
parser.add_argument('--lr_decay_rate', type=float, default=0.7,
                    help='learning rate rate for every decay [default: 0.7]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
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
parser.add_argument('--log', help='Log to FILE in save folder; use - for stdout (default is log.txt)', metavar='FILE',
                    default='log.txt')
parser.add_argument('--num_point', help='downsample number before feed to net', type=int, default=8192)
parser.add_argument('--step_val', help='downsample number before feed to net', type=int, default=200000)
parser.add_argument('--model_name', '-m', default='pointsemantic_cross', help='Model to use')
parser.add_argument('--use_normals', action='store_true')
parser.add_argument("--train_set", default="train", help="train, train_full")
parser.add_argument("--dataset_name", default="semantic_plus_npm", help="npm, semantic")
args = parser.parse_args()

GPU_ID = args.gpu_id
MAX_EPOCH = args.max_epoch
BATCH_SIZE_TRAIN = args.batch_size_train
BATCH_SIZE_VAL = args.batch_size_val
STEP_VAL = args.step_val
TRAIN_AUGMENTATION = get_augmentations_from_list(args.augmentation, upright_axis=args.upright_axis)
RESUME_MODEL = args.resume_model
SUMMARY_LOG_DIR = args.summary_log_dir
MODEL_OPTIMIZER = args.optimizer
INIT_LEARNING_RATE = args.init_learning_rate
LR_DECAY_STEP = args.lr_decay_step
LR_DECAY_RATE = args.lr_decay_rate
MODEL_NAME = args.model_name
LOG = args.log
NUM_POINT = args.num_point
USE_NORMALS = args.use_normals
DATASET_NAME = args.dataset_name

train_augmentations = get_augmentations_from_list(TRAIN_AUGMENTATION)

SAVE_FOLDER = 'train_log/' + MODEL_NAME + '_' + DATASET_NAME
root_folder = SAVE_FOLDER
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

PARAMS = json.loads(open("semantic_no_color.json").read())
# Import dataset
TRAIN_DATASET = SemanticDataset(
    num_points_per_sample=NUM_POINT,
    split=args.train_set,
    box_size_x=PARAMS["box_size_x"],
    box_size_y=PARAMS["box_size_y"],
    use_color=PARAMS["use_color"],
    use_geometry=PARAMS["use_geometry"],
    path=PARAMS["data_path"],
)
VALIDATION_DATASET = SemanticDataset(
    num_points_per_sample=NUM_POINT,
    split="validation",
    box_size_x=PARAMS["box_size_x"],
    box_size_y=PARAMS["box_size_y"],
    use_color=PARAMS["use_color"],
    use_geometry=PARAMS["use_geometry"],
    path=PARAMS["data_path"],
)
NPM_PARAMS = json.loads(open("npm.json").read())
NPM_VALIDATION_DATASET = NpmDataset(
    num_points_per_sample=NUM_POINT,
    split="validation",
    box_size_x=NPM_PARAMS["box_size_x"],
    box_size_y=NPM_PARAMS["box_size_y"],
    use_geometry=NPM_PARAMS["use_geometry"],
    path=NPM_PARAMS["data_path"],
)

num_classes = TRAIN_DATASET.num_classes
label_weights = TRAIN_DATASET.label_weights
# start logging
LOG_FOUT = open(os.path.join(root_folder, LOG), 'w')
EPOCH_CNT = 0
LOG_FOUT.write(str(args) + '\n')

print('PID:', os.getpid())

print(args)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def update_progress(progress):
    """
    Displays or updates a console progress bar
    Args:
        progress: A float between 0 and 1. Any int will be converted to a float.
                  A value under 0 represents a 'halt'.
                  A value at 1 or bigger represents 100%
    """
    bar_length = 10  # Modify this to change the length of the progress bar
    if isinstance(progress, int):
        progress = round(float(progress), 2)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    text = "\rProgress: [{}] {}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100
    )
    sys.stdout.write(text)
    sys.stdout.flush()


def get_batch(split):
    np.random.seed()
    if split == "train":
        return TRAIN_DATASET.sample_batch_in_all_files(BATCH_SIZE_TRAIN,
                                                       augment=True), NPM_VALIDATION_DATASET.sample_batch_in_all_files(
            BATCH_SIZE_TRAIN, augment=True)
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
    print("we have %d batches for train and %d batches for validation in one epoch" %
          (num_train_batches, num_validation_batches))
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


def train_one_epoch(stack, scheduler, model, criterion, device, train_writer):
    global EPOCH_CNT
    num_batches = TRAIN_DATASET.get_num_batches(BATCH_SIZE_TRAIN)

    log_string(str(datetime.now()))
    update_progress(0)
    # Reset metrics
    loss_sum = 0
    confusion_matrix = metric.ConfusionMatrix(num_classes)

    # Train over num_batches batches
    for batch_idx in range(num_batches):
        # Refill more batches if empty
        progress = float(batch_idx) / float(num_batches)
        update_progress(round(progress, 2))
        (batch_data, batch_label, batch_weights), (npm_batch_data, _, _) = stack.get()

        # Get predicted labels
        points_tensor = torch.from_numpy(batch_data).to(device, dtype=torch.float32)  # (B, sample_num, 3)
        npm_points_tensor = torch.from_numpy(npm_batch_data).to(device, dtype=torch.float32)  # (B, sample_num, 3)
        batch_label_tensor = torch.from_numpy(batch_label).to(device, dtype=torch.long)  # (B, sample_num)
        scheduler.optimizer.zero_grad()
        model = model.train()
        points_prob, npm_reconstructed = run_model(model, points_tensor, PARAMS, MODEL_NAME, another_input=npm_points_tensor)  # (B, sample_num, num_classes), (B, sample_num, 3)
        batch_loss = criterion(points_prob, batch_label_tensor, npm_reconstructed, npm_points_tensor)
        _, points_pred = torch.max(points_prob, dim=2)  # (B, sample_num)
        batch_loss.backward()
        scheduler.optimizer.step()
        scheduler.step()

        # Update metrics
        pred_val = points_pred.cpu().numpy()
        for i in range(len(pred_val)):
            for j in range(len(pred_val[i])):
                confusion_matrix.increment(batch_label[i][j], pred_val[i][j])
        loss_sum += batch_loss.cpu().detach().numpy()
    update_progress(1)
    EPOCH_CNT += 1
    log_string("mean loss: %f" % (loss_sum / float(num_batches)))
    log_string("Overall accuracy : %f" % (confusion_matrix.get_accuracy()))
    log_string("Average IoU : %f" % (confusion_matrix.get_mean_iou()))
    train_writer.add_scalar("%s mean loss" % DATASET_NAME, loss_sum / float(num_batches), EPOCH_CNT)
    train_writer.add_scalar("%s overall accuracy" % DATASET_NAME, confusion_matrix.get_accuracy(), EPOCH_CNT)
    train_writer.add_scalar("%s average IoU" % DATASET_NAME, confusion_matrix.get_mean_iou(), EPOCH_CNT)
    iou_per_class = confusion_matrix.get_per_class_ious()
    iou_per_class = [0] + iou_per_class  # label 0 is ignored
    for i in range(1, num_classes):
        log_string("IoU of %s : %f" % (TRAIN_DATASET.labels_names[i], iou_per_class[i]))


def eval_one_epoch(stack, model, criterion, device, val_writer):
    num_batches = VALIDATION_DATASET.get_num_batches(BATCH_SIZE_VAL)

    # Reset metrics
    loss_sum = 0
    confusion_matrix = metric.ConfusionMatrix(num_classes)

    log_string(str(datetime.now()))

    log_string("---- EPOCH %03d EVALUATION ----" % (EPOCH_CNT))

    update_progress(0)

    for batch_idx in range(num_batches):
        progress = float(batch_idx) / float(num_batches)
        update_progress(round(progress, 2))
        batch_data, batch_label, batch_weights = stack.get()

        # Get predicted labels
        points_tensor = torch.from_numpy(batch_data).to(device, dtype=torch.float32)  # (B, sample_num, 3)
        batch_label_tensor = torch.from_numpy(batch_label).to(device, dtype=torch.long)  # (B, sample_num)
        model = model.eval()
        with torch.no_grad():
            points_prob, reconstructed = run_model(model, points_tensor, PARAMS, MODEL_NAME,
                                                   another_input=points_tensor)  # (B, sample_num, num_classes), (B, sample_num, 3)
            batch_loss = criterion(points_prob, batch_label_tensor, reconstructed, points_tensor)
        _, points_pred = torch.max(points_prob, dim=2)  # (B, sample_num)

        # Update metrics
        pred_val = points_pred.cpu().numpy()
        for i in range(len(pred_val)):
            for j in range(len(pred_val[i])):
                confusion_matrix.increment(batch_label[i][j], pred_val[i][j])
        loss_sum += batch_loss.cpu().numpy()

    update_progress(1)

    iou_per_class = confusion_matrix.get_per_class_ious()

    # Display metrics
    log_string("mean loss: %f" % (loss_sum / float(num_batches)))
    log_string("Overall accuracy : %f" % (confusion_matrix.get_accuracy()))
    log_string("Average IoU : %f" % (confusion_matrix.get_mean_iou()))
    val_writer.add_scalar("%s mean loss" % DATASET_NAME, loss_sum / float(num_batches), EPOCH_CNT)
    val_writer.add_scalar("%s overall accuracy" % DATASET_NAME, confusion_matrix.get_accuracy(), EPOCH_CNT)
    val_writer.add_scalar("%s average IoU" % DATASET_NAME, confusion_matrix.get_mean_iou(), EPOCH_CNT)
    iou_per_class = [0] + iou_per_class  # label 0 is ignored
    for i in range(1, num_classes):
        log_string(
            "IoU of %s : %f" % (VALIDATION_DATASET.labels_names[i], iou_per_class[i])
        )

    return confusion_matrix.get_accuracy()


def train():
    global EPOCH_CNT
    os.makedirs(os.path.join(root_folder, SUMMARY_LOG_DIR), exist_ok=True)
    train_writer = SummaryWriter(os.path.join(root_folder, SUMMARY_LOG_DIR, 'train'))
    val_writer = SummaryWriter(os.path.join(root_folder, SUMMARY_LOG_DIR, 'val'))

    # set model and criterion
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % GPU_ID)
    else:
        raise ValueError("GPU not found!")
    label_weights_tensor = torch.from_numpy(label_weights).to(device)
    model, criterion = select_model(MODEL_NAME, num_classes, PARAMS, weights=label_weights_tensor)
    model = model.to(device)
    criterion = criterion.to(device)

    if MODEL_OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), INIT_LEARNING_RATE)
    elif MODEL_OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), INIT_LEARNING_RATE)
    else:
        optimizer = None
        exit(0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_DECAY_STEP, gamma=LR_DECAY_RATE)

    if len(RESUME_MODEL) > 0:
        resume_path = os.path.join(root_folder, RESUME_MODEL)
        print("Resuming From ", resume_path)
        checkpoint = torch.load(resume_path)
        saved_state_dict = checkpoint['state_dict']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(saved_state_dict)
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
    EPOCH_CNT = start_epoch
    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    parameter_num = np.sum([np.prod(list(v.shape)) for v in model.parameters()])
    log_string('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    # start training
    stacker, stack_validation, stack_train = init_stacking()
    best_acc = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        print("in epoch", epoch)
        print("max_epoch", MAX_EPOCH)

        log_string("**** EPOCH %03d ****" % (epoch + 1))
        sys.stdout.flush()

        # Train one epoch
        train_one_epoch(stack_train, scheduler, model, criterion, device, train_writer)
        # Evaluate, save, and compute the accuracy
        if epoch % 5 == 0:
            acc = eval_one_epoch(stack_validation, model, criterion, device, val_writer)
            save_path = os.path.join(root_folder, 'checkpoint_epoch%d_acc%.2f.tar' % (epoch, acc))
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_path)
                log_string("Model saved in file: %s" % save_path)
                print("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_path)
                log_string("Model saved in file: %s" % save_path)
                print("Model saved in file: %s" % save_path)

    # Kill the process, close the file and exit
    stacker.terminate()
    LOG_FOUT.close()
    sys.exit()


if __name__ == "__main__":
    train()
