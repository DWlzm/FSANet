import torch
from thop import profile, clever_format

from FSANet import FSANet 


model = FSANet() 
model.eval()

input_tensor = torch.randn(1, 3, 256, 256) 


macs, params = profile(model, inputs=(input_tensor, ))


flops, params_readable = clever_format([macs * 2, params], "%.3f")

print(f"Params: {params_readable}")
print(f"FLOPs: {flops}")