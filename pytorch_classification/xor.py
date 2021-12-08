
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

model = load_checkpoint('U:\\tianma\dataset\weights\mobilenetv2\epoch_7000.pth')
f = './xor7000.pt'
#f.replace("pth", "xorpt")

example = torch.rand(1, 3, 224, 224)


traced_script_module = torch.jit.trace(model, example)

traced_script_module.save('7000.pt')
output = traced_script_module(torch.ones(1, 3, 224, 224))

print(output)

## 模型加密
with open(f, 'wb') as f1:
    buffer = io.BytesIO()
    torch.jit.save(traced_script_module, buffer)
    from itertools import cycle
    key = '123'
    buffer1 = buffer.getvalue()
    buffer2 = bytes([a ^ ord(b) for (a,b) in zip(buffer1,cycle(key))])
    # 测试
    print(len(buffer1))
    print(b"Origin: " + buffer1[0:5] + b", " + buffer1[-6:-1])
    print(b"Encoder: " + buffer2[0:5] + b", " + buffer2[-6:-1])



    f1.write(buffer2)


