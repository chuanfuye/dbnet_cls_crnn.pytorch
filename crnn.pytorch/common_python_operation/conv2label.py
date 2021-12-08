# -*- coding: utf-8 -*-
# import os
# 字典格式转换
import time

import chardet
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

char_set = open('common_python_operation/label.txt', 'r', encoding='utf-8').readlines()
char_set = ''.join([ch.strip('\n') for ch in char_set])
n_class = len(char_set)
char_set2 = open('common_python_operation/label2.txt', 'w', encoding='utf-8')
char_set2.write(char_set)

