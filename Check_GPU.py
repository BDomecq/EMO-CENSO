import torch

print("GPU Available: ", torch.cuda.is_available())
print("GPU number: ", torch.cuda.device_count())



