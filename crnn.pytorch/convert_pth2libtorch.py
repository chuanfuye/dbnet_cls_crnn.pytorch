# -*- coding: utf-8 -*-


import argparse
import os
import sys

import torch

import lib.models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--gpus',type = str,default='0',help='gradient clipping value,default = 0')
parser.add_argument('--imageH',type = int,default=32,help='height of input image')
parser.add_argument('--imageW',type = int,default=640,help='width of input image')
parser.add_argument('--imageChannel',type = int,default=3,help='input image channel number')
parser.add_argument('--pthModel',type = str,default='',help='pth model file name')
parser.add_argument('--libModel',type = str,default='',help='to libtorch model file name')

args = parser.parse_args()

# 转换模型
def convert_model2device2(state_model_file,device):
    model = crnn.CRNN(32,3,5992,256)
    checkpoint = torch.load(state_model_file)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    print("load ok")
    print(model)
    model.eval()
    cpu_name = name +'.'+(device if device =="cpu" else "gpu")
    model =model.to(device)
    example = torch.ones(1,3,32,160).to(device)
    traced_module = torch.jit.trace(model,example)
    traced_module.save(cpu_name)
    print("saved "+cpu_name)

def convert_model2device(state_model_file,name, device):
    model = crnn.CRNN(32,3,89,256)
    checkpoint = torch.load(state_model_file)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    print("load ok")
    print(model)
    model.eval()
    name = name.split(".")[0] + '_' + (device if device =="cpu" else "gpu") + ".pt"
    model =model.to(device)
    example = torch.ones(1,args.imageChannel,args.imageH,args.imageW).to(device)
    traced_module = torch.jit.trace(model,example)
    traced_module.save(name)
    print("saved "+ name)


if __name__ == '__main__':
    print("convert start......")

    state_model_file = 'output/icdar2015/crnn/2021-11-19-07-01/checkpoints/checkpoint_579_acc_0.3088.pth'
    name= "test_crnn.pt"
    convert_model2device(state_model_file, name,'cuda')

    print("convert ok !!!")
