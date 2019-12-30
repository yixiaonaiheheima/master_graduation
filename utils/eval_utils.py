# *_*coding:utf-8 *_*
import os
import open3d
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import datetime
import pandas as pd
import torch.nn.functional as F


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


# it not correct visualization, I think...
def show_example(x, y, x_reconstruction, y_pred, save_dir, figname):
    x = x.squeeze().cpu().data.numpy()  # (B, 3, N)
    x = x.permute(0, 2, 1)  # (B, N, 3)
    y = y.cpu().data.numpy()  # (B, N)
    x_reconstruction = x_reconstruction.squeeze().cpu().data.numpy()
    _, y_pred = torch.max(y_pred, -1)  # (B, N)
    y_pred = y_pred.cpu().data.numpy()  # (B, N)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x, cmap='Greys')
    ax[0].set_title('Input: %d' % y)
    ax[1].imshow(x_reconstruction, cmap='Greys')
    ax[1].set_title('Output: %d' % y_pred)
    plt.savefig(save_dir + figname + '.png')


def save_checkpoint(epoch, train_accuracy, test_accuracy, model, optimizer, path,modelnet='checkpoint'):
    savepath  = path + '/%s-%f-%04d.pth' % (modelnet,test_accuracy, epoch)
    state = {
        'epoch': epoch,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)


def test(model, loader):
    mean_correct = []
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    return np.mean(mean_correct)


def compute_cat_iou(pred,target,iou_tabel):
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]
        batch_target = target[j]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list


def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def test_partseg(model, loader, catdict, num_classes = 50,forpointnet2=False):
    ''' catdict = {0:Airplane, 1:Airplane, ...49:Table} '''
    iou_tabel = np.zeros((len(catdict),3))
    iou_list = []
    metrics = defaultdict(lambda:list())
    hist_acc = []
    # mean_correct = []
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point,_= points.size()
        points, label, target, norm_plt = Variable(points.float()),Variable(label.long()), Variable(target.long()),Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(), label.squeeze().cuda(), target.cuda(), norm_plt.cuda()
        if forpointnet2:
            seg_pred = model(points, norm_plt, to_categorical(label, 16))
        else:
            labels_pred, seg_pred, _  = model(points,to_categorical(label,16))
            # labels_pred_choice = labels_pred.data.max(1)[1]
            # labels_correct = labels_pred_choice.eq(label.long().data).cpu().sum()
            # mean_correct.append(labels_correct.item() / float(points.size()[0]))
        # print(pred.size())
        iou_tabel, iou = compute_cat_iou(seg_pred,target,iou_tabel)
        iou_list+=iou
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        seg_pred = seg_pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['inctance_avg_iou'] = np.mean(iou_list)
    # metrics['label_accuracy'] = np.mean(mean_correct)
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    metrics['class_avg_iou'] = np.mean(cat_iou)

    return metrics, hist_acc, cat_iou


def test_semseg(model, loader, catdict, num_classes = 13, pointnet2=False):
    iou_tabel = np.zeros((len(catdict),3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        points, target = Variable(points.float()), Variable(target.long())
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        if pointnet2:
            pred = model(points[:, :3, :], points[:, 3:, :])
        else:
            pred, _ = model(points)
        # print(pred.size())
        iou_tabel, iou_list = compute_cat_iou(pred,target,iou_tabel)
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    # print(iou_tabel)
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou


def compute_avg_curve(y, n_points_avg):
    avg_kernel = np.ones((n_points_avg,)) / n_points_avg
    rolling_mean = np.convolve(y, avg_kernel, mode='valid')
    return rolling_mean


def plot_loss_curve(history, n_points_avg, n_points_plot, save_dir):
    curve = np.asarray(history['loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-g')

    curve = np.asarray(history['margin_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-b')

    curve = np.asarray(history['reconstruction_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-r')

    plt.legend(['Total Loss', 'Margin Loss', 'Reconstruction Loss'])
    plt.savefig(save_dir + '/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M')) + '_total_result.png')
    plt.close()


def plot_acc_curve(total_train_acc, total_test_acc, save_dir):
    plt.plot(total_train_acc, '-b',label = 'train_acc')
    plt.plot(total_test_acc, '-r',label = 'test_acc')
    plt.legend()
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('Accuracy of training and test')
    plt.savefig(save_dir +'/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'_total_acc.png')
    plt.close()


def show_point_cloud(tuple, seg_label=[], title=None):
    import matplotlib.pyplot as plt
    if not seg_label:
        x = [x[0] for x in tuple]
        y = [y[1] for y in tuple]
        z = [z[2] for z in tuple]
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    else:
        category = list(np.unique(seg_label))
        color = ['b', 'r', 'g', 'y', 'w', 'b', 'p']
        ax = plt.subplot(111, projection='3d')
        for categ_index in range(len(category)):
            tuple_seg = tuple[seg_label == category[categ_index]]
            x = [x[0] for x in tuple_seg]
            y = [y[1] for y in tuple_seg]
            z = [z[2] for z in tuple_seg]
            ax.scatter(x, y, z, c=color[categ_index], cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    plt.title(title)
    plt.show()


def convert2new(npm_labels, semantic_labels):
    """
    convert semantic label and npm label to new class. We set new classes by 0: unlabeled, 1: ground,
    2: nature, 3: building, 4: hardscape 5: cars
    :param npm_labels: (N,) npm  labels available by dataset
    :param semantic_labels: (N,) semantic  labels availabel by model
    :return: (n,), (n,)
    """
    semantic2new = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 4, 7: 0, 8: 5}
    npm2new = {0: 0, 1: 1, 2: 3, 3: 4, 4: 4, 5: 4, 6: 4, 7: 0, 8: 5, 9: 2}
    npm_shape = npm_labels.shape
    semantic_shape = semantic_labels.shape
    assert(npm_shape == semantic_shape)
    npm_labels = npm_labels.flatten()
    semantic_labels = semantic_labels.flatten()
    N = len(npm_labels)
    for i in range(N):
        npm_labels[i] = npm2new[npm_labels[i]]
        semantic_labels[i] = semantic2new[semantic_labels[i]]
    npm_labels = np.reshape(npm_labels, npm_shape)
    semantic_labels = np.reshape(semantic_labels, npm_shape)
    return npm_labels, semantic_labels


def semantic2common(semantic_labels):
    """
    convert semantic label to common class. We set common classes by 0: unlabeled, 1: ground,
    2: nature, 3: building, 4: hardscape 5: cars
    :param semantic_labels: (N,) semantic  labels availabel by model
    :return: (n,), (n,)
    """
    semantic2new = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 4, 7: 0, 8: 5}
    semantic_shape = semantic_labels.shape
    semantic_labels = semantic_labels.flatten()
    N = len(semantic_labels)
    for i in range(N):
        semantic_labels[i] = semantic2new[semantic_labels[i]]
    semantic_labels = np.reshape(semantic_labels, semantic_shape)
    return semantic_labels


def npm2common(npm_labels):
    """
    convert semantic label to common class. We set common classes by 0: unlabeled, 1: ground,
    2: nature, 3: building, 4: hardscape 5: cars
    :param npm_labels: (N,) npm  labels available by dataset
    :return: (n,), (n,)
    """
    npm2new = {0: 0, 1: 1, 2: 3, 3: 4, 4: 4, 5: 4, 6: 4, 7: 0, 8: 5, 9: 2}
    npm_shape = npm_labels.shape
    npm_labels = npm_labels.flatten()
    N = len(npm_labels)
    for i in range(N):
        npm_labels[i] = npm2new[npm_labels[i]]
    npm_labels = np.reshape(npm_labels, npm_shape)
    return npm_labels


def _2common(labels, dataset_name):
    if dataset_name == 'semantic':
        return semantic2common(labels)
    elif dataset_name == 'npm':
        return npm2common(labels)
    else:
        raise ValueError("can't not recognize dataset name %s" % dataset_name)


if __name__ == '__main__':
    pcd_path = '../Lille2.pcd'
    label_path = '../Lille2.labels'
    pcd_file = open3d.io.read_point_cloud(pcd_path)
    cloud = np.asarray(pcd_file.points, dtype=np.float32)
    # label = np.loadtxt(label_path, dtype=np.int64)
    show_point_cloud(cloud)

    print('done')
