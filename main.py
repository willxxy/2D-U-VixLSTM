from twoDUVixLSTM import UVixLSTM
import torch

class_num = 1
input_tensor = torch.randn(1, 1, 96, 96)

model = UVixLSTM(class_num)

x = model(input_tensor)
print(x.shape) # torch.Size([1, 1, 96, 96])