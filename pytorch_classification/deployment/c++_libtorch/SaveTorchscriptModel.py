# -*- coding:utf-8 -*-
# @time :2020/8/28
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com


import torch
import io

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

model = load_checkpoint('U:\\tianma\dataset\weights\mobilenetv2\epoch_500.pth')
f = 'U:\\tianma\dataset\weights\mobilenetv2\epoch_500.xorpt'

example = torch.rand(1, 3, 224, 224)


traced_script_module = torch.jit.trace(model, example)

traced_script_module.save('./mobilenetv2_500.pt')
output = traced_script_module(torch.ones(1, 3, 224, 224))

print(output)

