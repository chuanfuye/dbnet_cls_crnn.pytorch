import torch
import lib.models.crnn as crnn

model = crnn.CRNN(32,3,5992,256)
checkpoint = torch.load("checkpoint_best_color.pth")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

batch_size = 1
dummy_input = torch.randn(batch_size,3,32,160)

#print(dummy_input)
input_names = [ "input"]
output_names = [ "output" ]
torch.onnx.export(model, dummy_input,"crnn.onnx",verbose=True,input_names=input_names, output_names=output_names,opset_version=11)
