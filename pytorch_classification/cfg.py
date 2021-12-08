# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
import os
home = os.path.expanduser('~')
#  数据集的类别,可修改
NUM_CLASSES = 2

#  训练时batch的大小
BATCH_SIZE = 8

#  网络默认输入图像的大小
INPUT_SIZE = 224

# 训练最多的epoch
MAX_EPOCH = 10000

# 使用gpu的个数
GPUS = 1

# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0

# 权重衰减和动量
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# 初始学习率
LR = 1e-3

# 采用的模型名称，可替换扩展，添加imagenet的预训练模型
model_name = 'resnet50'

from models import Resnet50, Resnet101, Resnext101_32x8d,Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, Efficientnet, Resnext101_32x32d, Resnext101_32x48d
MODEL_NAMES = {
    'resnext101_32x8d': Resnext101_32x8d,
    'resnext101_32x16d': Resnext101_32x16d,
    'resnext101_32x48d': Resnext101_32x48d,
    'resnext101_32x32d': Resnext101_32x32d,
    'resnet50': Resnet50,
    'resnet101': Resnet101,
    'densenet121': Densenet121,
    'densenet169': Densenet169,
    'mobilenetv2': Mobilenetv2,
    'efficientnet-b7': Efficientnet,
    'efficientnet-b8': Efficientnet
}

# 基础数据
BASE = 'U:\\tianma\\dataset\\'


# 训练好模型的保存位置
SAVE_FOLDER = BASE + 'weights/'

# 数据集的存放位置
TRAIN_LABEL_DIR = BASE + 'train.txt'
VAL_LABEL_DIR = BASE + 'val.txt'
TEST_LABEL_DIR = BASE + 'test.txt'


# 训练完成，权重文件的保存路径,默认保存在trained_model下
TRAINED_MODEL = BASE + 'weights/Resnet50/epoch_900.pth'




