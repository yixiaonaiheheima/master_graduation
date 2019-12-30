import argparse
import os
import json
import numpy as np
import open3d
import time
import torch
from data.semantic_dataset import SemanticDataset
from data.npm_dataset import NpmDataset
from utils.metric import ConfusionMatrix
from utils.model_util import select_model, run_model
from utils.eval_utils import _2common
from utils.point_cloud_util import _label_to_colors


# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1, help='gpu id for network')
parser.add_argument("--num_samples", type=int, default=1000, help="# samples, each contains num_point points_centered")
parser.add_argument("--resume_model", default="/home/yss/sda1/yzl/yzl_graduation/train_log/pointnet2_semantic_row7/checkpoint_epoch70_acc0.8707133112726985.tar", help="restore checkpoint file storing model parameters")
parser.add_argument("--config_file", default="semantic.json",
                    help="config file path, it should same with that during traing")
parser.add_argument("--set", default="validation", help="train, validation, test")
parser.add_argument('--num_point', help='downsample number before feed to net', type=int, default=8192)
parser.add_argument('--model_name', '-m', help='Model to use', required=True)
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch Size for prediction [default: 32]')
parser.add_argument('--from_dataset', default='semantic', help='which dataset the model is trained from')
parser.add_argument('--to_dataset', default='semantic', help='which dataset to predict')
flags = parser.parse_args()


if __name__ == "__main__":
    np.random.seed(0)
    hyper_params = json.loads(open(flags.config_file).read())

    # Create output dir
    sub_folder = flags.model_name + '_' + flags.from_dataset + '2' + flags.to_dataset
    output_dir = os.path.join("result", "sparse", sub_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    if flags.to_dataset == 'semantic':
        dataset = SemanticDataset(
            num_points_per_sample=flags.num_point,
            split=flags.set,
            box_size_x=hyper_params["box_size_x"],
            box_size_y=hyper_params["box_size_y"],
            use_color=hyper_params["use_color"],
            use_geometry=hyper_params['use_geometry'],
            path=hyper_params["data_path"],
        )
    elif flags.to_dataset == 'npm':
        dataset = NpmDataset(
            num_points_per_sample=flags.num_point,
            split=flags.set,
            box_size_x=hyper_params["box_size_x"],
            box_size_y=hyper_params["box_size_y"],
            use_geometry=hyper_params['use_geometry'],
            path=hyper_params["data_path"],
        )
    else:
        print("dataset error")
        raise ValueError

    # Model
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % flags.gpu_id)
    else:
        raise ValueError("GPU not found!")
    batch_size = flags.batch_size
    if flags.from_dataset == 'semantic' and flags.to_dataset == 'npm':
        num_classes = 6
        classes_in_model = 9
    else:
        num_classes = dataset.num_classes
        classes_in_model = num_classes
    # load model
    resume_path = flags.resume_model
    model = select_model(flags.model_name, classes_in_model, hyper_params)[0]
    model = model.to(device)
    print("Resuming From ", resume_path)
    checkpoint = torch.load(resume_path)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    # Process each file
    cm = ConfusionMatrix(num_classes)
    common_cm = ConfusionMatrix(6)
    model = model.eval()
    for file_data in dataset.list_file_data:
        print("Processing {}".format(file_data.file_path_without_ext))

        # Predict for num_samples times
        points_collector = []
        pd_labels_collector = []
        pd_prob_collector = []
        pd_common_labels_collector = []

        # If flags.num_samples < batch_size, will predict one batch
        for batch_index in range(int(np.ceil(flags.num_samples / batch_size))):
            current_batch_size = min(batch_size, flags.num_samples - batch_index * batch_size)

            # Get data
            if flags.to_dataset == 'semantic':
                points_centered, points, gt_labels, colors, geometry = file_data.sample_batch(
                    batch_size=current_batch_size,
                    num_points_per_sample=flags.num_point,
                )
            else:
                points_centered, points, gt_labels, geometry = file_data.sample_batch(
                    batch_size=current_batch_size,
                    num_points_per_sample=flags.num_point,
                )

            data_list = [points_centered]
            # only semantic3d dataset can set 'use_color' 1
            if hyper_params['use_color']:
                data_list.append(colors)
            if hyper_params['use_geometry']:
                data_list.append(geometry)
            point_cloud = np.concatenate(data_list, axis=-1)

            # Predict
            s = time.time()
            input_tensor = torch.from_numpy(point_cloud).to(device, dtype=torch.float32)  # (current_batch_size, N, 3)
            with torch.no_grad():
                pd_prob = run_model(model, input_tensor, hyper_params, flags.model_name)  # (current_batch_size, N)
            _, pd_labels = torch.max(pd_prob, dim=2)  # (B, N)
            pd_prob = pd_prob.cpu().numpy()
            pd_labels = pd_labels.cpu().numpy()
            print("Batch size: {}, time: {}".format(current_batch_size, time.time() - s))

            common_gt = _2common(gt_labels, flags.to_dataset)  # (B, N)
            common_pd = _2common(pd_labels, flags.from_dataset)  # (B, N)

            # Save to collector for file output
            points_collector.extend(points)  # (B, N, 3)
            pd_labels_collector.extend(pd_labels)  # (B, N)
            pd_common_labels_collector.extend(common_pd)  # (B, N)
            pd_prob_collector.extend(pd_prob)  # (B, N, num_classes)

            # Increment confusion matrix

            common_cm.increment_from_list(common_gt.flatten(), common_pd.flatten())
            if flags.from_dataset == flags.to_dataset:
                cm.increment_from_list(gt_labels.flatten(), pd_labels.flatten())

        # Save sparse point cloud and predicted labels
        file_prefix = os.path.basename(file_data.file_path_without_ext)

        sparse_points = np.array(points_collector).reshape((-1, 3))  # (B*N, 3)
        sparse_common_labels = np.array(pd_common_labels_collector).flatten()
        pcd_common = open3d.geometry.PointCloud()
        pcd_common.points = open3d.utility.Vector3dVector(sparse_points)
        pcd_common.colors = open3d.utility.Vector3dVector(_label_to_colors(sparse_common_labels))
        pcd_path = os.path.join(output_dir, file_prefix + "_common.pcd")
        open3d.io.write_point_cloud(pcd_path, pcd_common)
        print("Exported sparse common pcd to {}".format(pcd_path))

        pd_labels_path = os.path.join(output_dir, file_prefix + "_common.labels")
        np.savetxt(pd_labels_path, sparse_common_labels, fmt="%d")
        print("Exported sparse common labels to {}".format(pd_labels_path))

        sparse_prob = np.array(pd_prob_collector).astype(float).flatten()
        pd_probs_path = os.path.join(output_dir, file_prefix + ".prob")
        np.savetxt(pd_probs_path, sparse_prob, fmt="%f")
        print("Exported sparse probs to {}".format(pd_probs_path))

        # save original labels and visulize them if from_dataset is equal to to_dataset
        if flags.from_dataset == flags.to_dataset:
            sparse_labels = np.array(pd_labels_collector).astype(int).flatten()
            pcd_ori = open3d.geometry.PointCloud()
            pcd_ori.points = open3d.utility.Vector3dVector(sparse_points)
            pcd_ori.colors = open3d.utility.Vector3dVector(_label_to_colors(sparse_labels))
            pcd_ori_path = os.path.join(output_dir, file_prefix + ".pcd")
            open3d.io.write_point_cloud(pcd_ori_path, pcd_ori)
            print("Exported sparse pcd to {}".format(pcd_ori_path))

            pd_ori_labels_path = os.path.join(output_dir, file_prefix + ".labels")
            np.savetxt(pd_ori_labels_path, sparse_labels, fmt="%d")
            print("Exported sparse labels to {}".format(pd_ori_labels_path))

    print("the following is the result of common class:")
    common_cm.print_metrics()
    print("#" * 100)
    if flags.from_dataset == flags.to_dataset:
        print("the following is the result of original class:")
        cm.print_metrics()
