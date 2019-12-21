import numpy as np
import os
import h5py

# ours: nohup python train.py -s model/ours --model PointSemantic --no_timestamp_folder > ours.out 2>&1 &
# pointnet: nohup python train_others.py --gpu_id 1 --model pointnet2 --no_timestamp_folder > pointnet2.out 2>&1 &
# pointcnn: nohup python train_others.py --gpu_id 1 --model pointcnn --batch_size_train 12 --batch_size_val 12 --no_timestamp_folder > pointcnn.out 2>&1 &
a = 5

def test1():
    return 1, 2, 3 if a == 4 else None



print('done')
