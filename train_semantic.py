import os
import sys
import json
import datetime
import numpy as np
# import tensorflow as tf
import multiprocessing as mp
import argparse
import time
from datetime import datetime
# import dataset and metric
from data.semantic_dataset import SemanticDataset
import utils.metric as metric
# import model
from models.pointnet2 import PointNet2Seg

from models.loss import PointnetCriterion
import torch

# Two global arg collections
parser = argparse.ArgumentParser()
parser.add_argument("--train_set", default="train", help="train, train_full")
parser.add_argument("--config_file", default="semantic.json", help="config file path")
parser.add_argument('--resume_model', type=str, default='',
                    help='If present, restore checkpoint and resume training')

FLAGS = parser.parse_args()
PARAMS = json.loads(open(FLAGS.config_file).read())
os.makedirs(PARAMS["logdir"], exist_ok=True)

# Import dataset
TRAIN_DATASET = SemanticDataset(
    num_points_per_sample=PARAMS["num_point"],
    split=FLAGS.train_set,
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
NUM_CLASSES = TRAIN_DATASET.num_classes

# Start logging
LOG_FOUT = open(os.path.join(PARAMS["logdir"], "log_train.txt"), "w")
EPOCH_CNT = 0


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
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
    barLength = 10  # Modify this to change the length of the progress bar
    if isinstance(progress, int):
        progress = round(float(progress), 2)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(barLength * progress))
    text = "\rProgress: [{}] {}%".format(
        "#" * block + "-" * (barLength - block), progress * 100
    )
    sys.stdout.write(text)
    sys.stdout.flush()


def get_batch(split):
    np.random.seed()
    if split == "train":
        return TRAIN_DATASET.sample_batch_in_all_files(
            PARAMS["batch_size"], augment=True
        )
    else:
        return VALIDATION_DATASET.sample_batch_in_all_files(
            PARAMS["batch_size"], augment=False
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
    num_train_batches = TRAIN_DATASET.get_num_batches(PARAMS["batch_size"])
    num_validation_batches = VALIDATION_DATASET.get_num_batches(
        PARAMS["batch_size"]
    )
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


def run_model(model, P):
    """

    :param model:
    :param P: tensor(B, N, C)
    :return:
    """
    points = P[:, :, :3]
    if PARAMS['use_color']:
        features = P[:, :, 3:]
    else:
        features = points
    if PARAMS['use_color']:
        res, _ = model(P.permute(0, 2, 1))
    else:
        res, _ = model(points.permute(0, 2, 1))
    return res


def train_one_epoch(stack, scheduler, model, criterion, device):
    """Train one epoch

    Args:
        sess (tf.Session): the session to evaluate Tensors and ops
        ops (dict of tf.Operation): contain multiple operation mapped with with strings
        train_writer (tf.FileSaver): enable to log the training with TensorBoard
        compute_class_iou (bool): it takes time to compute the iou per class, so you can
                                  disable it here
    """

    num_batches = TRAIN_DATASET.get_num_batches(PARAMS["batch_size"])

    log_string(str(datetime.now()))
    update_progress(0)
    # Reset metrics
    loss_sum = 0
    confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

    # Train over num_batches batches
    for batch_idx in range(num_batches):
        # Refill more batches if empty
        progress = float(batch_idx) / float(num_batches)
        update_progress(round(progress, 2))
        batch_data, batch_label, batch_weights = stack.get()

        # Get predicted labels
        points_tensor = torch.from_numpy(batch_data).to(device, dtype=torch.float32)  # (B, sample_num, 3)
        batch_label_tensor = torch.from_numpy(batch_label).to(device, dtype=torch.long)  # (B, sample_num)
        scheduler.optimizer.zero_grad()
        model = model.train()
        points_prob = run_model(model, points_tensor)  # (B, sample_num, num_classes), (B, sample_num, 3)
        batch_loss = criterion(points_prob, batch_label_tensor)
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
    log_string("mean loss: %f" % (loss_sum / float(num_batches)))
    log_string("Overall accuracy : %f" % (confusion_matrix.get_accuracy()))
    log_string("Average IoU : %f" % (confusion_matrix.get_mean_iou()))
    iou_per_class = confusion_matrix.get_per_class_ious()
    iou_per_class = [0] + iou_per_class  # label 0 is ignored
    for i in range(1, NUM_CLASSES):
        log_string("IoU of %s : %f" % (TRAIN_DATASET.labels_names[i], iou_per_class[i]))


def eval_one_epoch(stack, scheduler, model, criterion, device):
    """Evaluate one epoch

    Args:
        sess (tf.Session): the session to evaluate tensors and operations
        ops (tf.Operation): the dict of operations
        validation_writer (tf.summary.FileWriter): enable to log the evaluation on TensorBoard

    Returns:
        float: the overall accuracy computed on the validationation set
    """

    global EPOCH_CNT

    num_batches = VALIDATION_DATASET.get_num_batches(PARAMS["batch_size"])

    # Reset metrics
    loss_sum = 0
    confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

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
            points_prob = run_model(model, points_tensor)  # (B, sample_num, num_classes), (B, sample_num, 3)
            batch_loss = criterion(points_prob, batch_label_tensor)
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
    iou_per_class = [0] + iou_per_class  # label 0 is ignored
    for i in range(1, NUM_CLASSES):
        log_string(
            "IoU of %s : %f" % (VALIDATION_DATASET.labels_names[i], iou_per_class[i])
        )

    EPOCH_CNT += 5
    return confusion_matrix.get_accuracy()


def train():
    """Train the model on a single GPU
    """
    assert(torch.cuda.is_available())
    device = torch.device("cuda:%s" % PARAMS['gpu'])
    model = PointNet2Seg(NUM_CLASSES, with_rgb=PARAMS['use_color'])
    criterion = PointnetCriterion()
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), PARAMS['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, PARAMS['decay_step'], gamma=PARAMS['learning_rate_decay_rate'])
    if len(FLAGS.resume_model) > 0:
        resume_path = os.path.join(PARAMS['logdir'], FLAGS.resume_model)
        print("Resuming From ", resume_path)
        checkpoint = torch.load(resume_path)
        saved_state_dict = checkpoint['state_dict']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(saved_state_dict)
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0

    stacker, stack_validation, stack_train = init_stacking()

    # Train for hyper_params["max_epoch"] epochs
    best_acc = 0
    for epoch in range(start_epoch, PARAMS["max_epoch"]):
        print("in epoch", epoch)
        print("max_epoch", PARAMS["max_epoch"])

        log_string("**** EPOCH %03d ****" % (epoch))
        sys.stdout.flush()

        # Train one epoch
        train_one_epoch(stack_train, scheduler, model, criterion, device)
        save_path = os.path.join(PARAMS['logdir'], 'checkpoint_epoch{}.tar'.format(epoch))
        # Evaluate, save, and compute the accuracy
        if epoch % 5 == 0:
            acc = eval_one_epoch(stack_validation, scheduler, model, criterion, device)
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_path)
                log_string("Model saved in file: %s_%f" % (save_path, acc))
                print("Model saved in file: %s" % save_path)

        # Save the variables to disk.
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
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
