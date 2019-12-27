import argparse
import os
import json
import numpy as np
import open3d
import time
import torch
from data.semantic_dataset import SemanticDataset
from utils.metric import ConfusionMatrix
from utils.model_util import select_model, run_model


# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id for network')
parser.add_argument("--num_samples", type=int, default=500, help="# samples, each contains num_point points_centered")
parser.add_argument("--resume_model", default="/home/yss/sda1/yzl/yzl_graduation/train_log/pointnet2_semantic_row2/checkpoint_epoch90.tar", help="restore checkpoint file storing model parameters")
parser.add_argument("--config_file", default="semantic.json",
                    help="config file path, it should same with that during traing")
parser.add_argument("--set", default="validation", help="train, validation, test")
parser.add_argument('--num_point', help='downsample number before feed to net', type=int, default=8192)
parser.add_argument('--model_name', '-m', help='Model to use', required=True)
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch Size for prediction [default: 32]')
flags = parser.parse_args()


if __name__ == "__main__":
    np.random.seed(0)
    hyper_params = json.loads(open(flags.config_file).read())

    # Create output dir
    output_dir = os.path.join("result", "sparse")
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    dataset = SemanticDataset(
        num_points_per_sample=flags.num_point,
        split=flags.set,
        box_size_x=hyper_params["box_size_x"],
        box_size_y=hyper_params["box_size_y"],
        use_color=hyper_params["use_color"],
        path=hyper_params["data_path"],
    )

    # Model
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % flags.gpu_id)
    else:
        raise ValueError("GPU not found!")
    batch_size = flags.batch_size
    num_classes = dataset.num_classes
    # load model
    resume_path = flags.resume_model
    model, _ = select_model(flags.model_name, num_classes, hyper_params)
    model = model.to(device)
    print("Resuming From ", resume_path)
    checkpoint = torch.load(resume_path)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    # Process each file
    cm = ConfusionMatrix(num_classes)
    model = model.eval()
    for semantic_file_data in dataset.list_file_data:
        print("Processing {}".format(semantic_file_data.file_path_without_ext))

        # Predict for num_samples times
        points_collector = []
        pd_labels_collector = []
        pd_prob_collector = []

        # If flags.num_samples < batch_size, will predict one batch
        for batch_index in range(int(np.ceil(flags.num_samples / batch_size))):
            current_batch_size = min(
                batch_size, flags.num_samples - batch_index * batch_size
            )

            # Get data
            points_centered, points, gt_labels, colors = semantic_file_data.sample_batch(
                batch_size=current_batch_size,
                num_points_per_sample=flags.num_point,
            )

            # (bs, 8192, 3) concat (bs, 8192, 3) -> (bs, 8192, 6)
            if hyper_params["use_color"]:
                points_centered_with_colors = np.concatenate(
                    (points_centered, colors), axis=-1
                )
            else:
                points_centered_with_colors = points_centered

            # Predict
            s = time.time()
            input_tensor = torch.from_numpy(points_centered_with_colors).to(device, dtype=torch.float32)  # (current_batch_size, N, 3)
            with torch.no_grad():
                pd_prob = run_model(model, input_tensor, hyper_params, flags.model_name)  # (current_batch_size, N)
            _, pd_labels = torch.max(pd_prob, dim=2)  # (B, N)
            pd_prob = pd_prob.cpu().numpy()
            pd_labels = pd_labels.cpu().numpy()
            print(
                "Batch size: {}, time: {}".format(current_batch_size, time.time() - s)
            )

            # Save to collector for file output
            points_collector.extend(points)  # (B, N, 3)
            pd_labels_collector.extend(pd_labels)  # (B, N)
            pd_prob_collector.extend(pd_prob)  # (B, N, num_classes)

            # Increment confusion matrix
            cm.increment_from_list(gt_labels.flatten(), pd_labels.flatten())

        # Save sparse point cloud and predicted labels
        file_prefix = os.path.basename(semantic_file_data.file_path_without_ext)

        sparse_points = np.array(points_collector).reshape((-1, 3))  # (B*N, 3)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(sparse_points)
        pcd_path = os.path.join(output_dir, file_prefix + ".pcd")
        open3d.io.write_point_cloud(pcd_path, pcd)
        print("Exported sparse pcd to {}".format(pcd_path))

        sparse_labels = np.array(pd_labels_collector).astype(int).flatten()
        pd_labels_path = os.path.join(output_dir, file_prefix + ".labels")
        np.savetxt(pd_labels_path, sparse_labels, fmt="%d")
        print("Exported sparse labels to {}".format(pd_labels_path))

        sparse_prob = np.array(pd_prob_collector).astype(float).flatten()
        pd_probs_path = os.path.join(output_dir, file_prefix + ".prob")
        np.savetxt(pd_probs_path, sparse_prob, fmt="%f")
        print("Exported sparse probs to {}".format(pd_probs_path))

    cm.print_metrics()
