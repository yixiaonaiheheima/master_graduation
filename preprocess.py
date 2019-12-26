import os
import subprocess
import shutil
import open3d
import sys
from data.semantic_dataset import all_file_prefixes
from data.npm_dataset import npm_all_file_prefixes
from plyfile import PlyData
import numpy as np
from utils.point_cloud_util import write_labels


def wc(file_name):
    out = subprocess.Popen(
        ["wc", "-l", file_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).communicate()[0]
    return int(out.partition(b" ")[0])


def prepend_line(file_name, line):
    with open(file_name, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip("\r\n") + "\n" + content)


def point_cloud_txt_to_pcd(raw_dir, file_prefix):
    # File names
    txt_file = os.path.join(raw_dir, file_prefix + ".txt")
    pts_file = os.path.join(raw_dir, file_prefix + ".pts")
    pcd_file = os.path.join(raw_dir, file_prefix + ".pcd")

    # Skip if already done
    if os.path.isfile(pcd_file):
        print("pcd {} exists, skipped".format(pcd_file))
        return

    # .txt to .pts
    # We could just prepend the line count, however, there are some intensity value
    # which are non-integers.
    print("[txt->pts]")
    print("txt: {}".format(txt_file))
    print("pts: {}".format(pts_file))
    first_line = str(wc(txt_file))

    with open(txt_file, "r") as txt_f, open(pts_file, "w") as pts_f:
        for line in txt_f:
            # x, y, z, i, r, g, b
            tokens = line.split()
            tokens[3] = str(int(float(tokens[3])))
            line = " ".join(tokens)
            pts_f.write(line + "\n")

    prepend_line(pts_file, first_line)

    # .pts -> .pcd
    print("[pts->pcd]")
    print("pts: {}".format(pts_file))
    print("pcd: {}".format(pcd_file))
    point_cloud = open3d.io.read_point_cloud(pts_file)
    open3d.io.write_point_cloud(pcd_file, point_cloud)
    os.remove(pts_file)


def point_cloud_ply_to_pcd(raw_dir, file_prefix):
    # File names
    ply_file = os.path.join(raw_dir, file_prefix + ".ply")
    pcd_file = os.path.join(raw_dir, file_prefix + ".pcd")
    label_file = os.path.join(raw_dir, file_prefix + ".labels")

    # Skip if already done
    if os.path.isfile(pcd_file):
        print("pcd {} exists, skipped".format(pcd_file))
        return

    # .pts -> .pcd
    columns = ["x", "y", "z", "reflectance"]
    ply_data = PlyData.read(ply_file)
    ply_data = ply_data.elements[0].data  # ndarray
    xyzi = np.array([ply_data[i] for i in columns]).transpose()  # (n, 4)
    if ('class', '<i4') in ply_data.dtype.descr:
        labels = ply_data['class']
    else:
        labels = None
    print("[ply->pcd]")
    print("ply: {}".format(ply_file))
    print("pcd: {}".format(pcd_file))
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(xyzi[:, :3])  # (n, 3)
    open3d.io.write_point_cloud(pcd_file, point_cloud)
    if labels is not None:
        write_labels(label_file, labels)


if __name__ == "__main__":
    # By default
    # raw data: "dataset/semantic_raw"
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # dataset_dir = os.path.join(current_dir, "dataset")
    # raw_dir = os.path.join(dataset_dir, "semantic_raw")
    #
    # for file_prefix in all_file_prefixes:
    #     point_cloud_txt_to_pcd(raw_dir, file_prefix)

    # npm preprocess
    dataset_dir = "/home/yss/sda1/yzl/Data"
    raw_dir = os.path.join(dataset_dir, "npm_raw")

    for file_prefix in npm_all_file_prefixes:
        point_cloud_ply_to_pcd(raw_dir, file_prefix)