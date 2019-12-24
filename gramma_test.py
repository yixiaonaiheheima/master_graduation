import numpy as np
import os
import h5py
import glob

# ours: nohup python train.py -s model/ours --model PointSemantic --no_timestamp_folder > ours.out 2>&1 &
# pointnet: nohup python train_others.py --gpu_id 1 --model pointnet2 --no_timestamp_folder > pointnet2.out 2>&1 &
# pointcnn: nohup python train_others.py --gpu_id 1 --model pointcnn --batch_size_train 12 --batch_size_val 12 --no_timestamp_folder > pointcnn.out 2>&1 &
# pointnet: nohup python train_one_dataset.py --gpu_id 1 --model pointnet --batch_size_train 24 --batch_size_val 24 --no_timestamp_folder > pointnet.out 2>&1 &
# pointnet2: nohup python train_one_dataset.py --gpu_id 0 --model pointnet2 --batch_size_train 12 --s
# batch_size_val 12 --no_timestamp_folder > pointnet2.out 2>&1 &
a = 5

root_folder = '/home/yss/sda1/yzl/Data/Semantic3D/val'
h5_files = glob.glob(root_folder + '/*.h5')
out_file = 'data_distribution.txt'
with open(out_file, 'w') as fid:
    for h5_name in h5_files:
        class_ratio = dict()
        for i in range(8):
            class_ratio[i] = 0.0
        # h5_name = 'sg27_station9_intensity_rgb_zero_13.h5'
        h5_dataframe = h5py.File(h5_name, 'r')
        # points = h5_dataframe['data'][...].astype(np.float32)
        # labels = h5_dataframe['label'][...].astype(np.int64)
        # point_nums = h5_dataframe['data_num'][...].astype(np.int32)
        labels_seg = h5_dataframe['label_seg'][...].astype(np.int64)
        unique, unique_count = np.unique(labels_seg, return_counts=True)
        all_count = np.sum(unique_count)
        count_norm = unique_count / all_count
        for class_idx in range(len(unique)):
            class_ratio[unique[class_idx]] = count_norm[class_idx]
        fid.write(h5_name)
        for i in range(8):
            fid.write('\t' + str(class_ratio[i]))
        fid.write('\n')
        print("count one file!")

print('done')
