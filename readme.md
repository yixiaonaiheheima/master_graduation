[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
# [激光雷达点云语义分割]() 

## 介绍

本代码仓为厦门大学2020届硕士毕业生余增磊的硕士毕业论文《激光雷达点云语义分割》的pytorch实现

## 安装

我推荐使用anaconda安装本文所需要的环境

1. 安装 [anaconda](https://www.anaconda.com/) Python版本选择3.6

2. 创建 `PBCNet` 的Conda环境: `conda create -n PBCNet python=3.6`.

3. 激活Conda环境: `conda activate PBCNet`.

4. 使用pip安装必要的包：`pip install -r requirements.txt`

4. 本文代码只在pytorch 1.5, cuda10.1的环境下进行过测试

## 数据集
本文已在[NPM3D](https://npm3d.fr//)和[Semantic3D](http://www.semantic3d.net/)数据集上进行测试

## 运行代码

### 数据预处理
- 将Semantic3D和NPM3D的点云数据统一转化成pcd格式方便后续处理
```
python preprocess.py --dataset_name semantic && python preprocess.py --dataset_name npm
```
- 将原始数据进行下采样
```
python downsample.py --dataset_name semantic && python downsample.py --dataset_name npm
```
- 计算局部几何信息
```
cd data
python add_geometry_to_downsample.py --dataset_name semantic && python add_geometry_to_downsample.py --dataset_name npm
cd ..
```

### 训练

- 在`NPM3D`上训练PBCNet: 
```
python train_one_dataest.py --model PBC_pure --dataset_name npm --config_file npm.json --batch_size_train 16 --batch_size_val 16
```
如果想使用局部几何信息，请修改npm.json中的`use_geometry`参数为1。

- 训练`NPM3D`到`Semantic3D`的跨源迁移模型
```
python train_cross.py --num_points 4096 --model PBC_atlas --batch_size_train 12 --batch_size_val 12 --dataset npm_plus_semantic
```
除了PBCNet，论文中设计的所有对比算法本代码仓均有实现，训练方法类似，下文同理。
### 验证
本文算法的预训练模型可通过[这里](https://drive.google.com/drive/folders/1CTlgSCjc80zZtg83IV2jZ442cazR1C_7?usp=sharing)下载, 请在predict.py的`--resume_model`参数中中设置合适的路径。
- semantic2semantic的同源语义分割测试:   
```
python predict.py --batch_size 16 --num_point 8192 --model_name PBC_pure --config_file semantic.json --from_dataset semantic --to_dataset semantic --set validation
```
- semantic2npm的同源语义分割测试: 
```
python predict.py --batch_size 12 --num_point 4096 --model_name PBC_folding --config_file npm.json --from_dataset semantic --to_dataset npm --set validation
```

### 提交到Benchmark
以提交到Semantic3D官网为例
- 得到Semantic3D测试集下采样后的稀疏点云的语义推断分数
```
python predict.py --batch_size 16 --num_point 8192 --model_name PBC_pure --config_file semantic.json --from_dataset semantic --to_dataset semantic --set test
```
- 插值成原始密集点云，请对代码的开头部分进行微调
```
python interpolate.py
```
- 重命名,请自行更改文件中的路径
```angular2html
python rename_semantic.py
```
最后将结果压缩成zip格式上传至官网即可。