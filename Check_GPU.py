import torch

print("GPU Available: ", torch.cuda.is_available())
print("GPU number: ", torch.cuda.device_count())


a = [100,150,200]
center = [250]

print("Cond: ", all(center[0] > coord for coord in a))
print("Max: ", max(a))


