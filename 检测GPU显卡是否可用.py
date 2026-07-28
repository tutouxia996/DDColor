import torch

# 1. 检测CUDA是否可用
print(f"CUDA是否可用：{torch.cuda.is_available()}")
# 2. 检测PyTorch编译是否带CUDA
print(f"PyTorch编译CUDA版本：{torch.version.cuda}")
# 3. 检测显卡数量/名称
if torch.cuda.is_available():
    print(f"GPU数量：{torch.cuda.device_count()}")
    print(f"GPU名称：{torch.cuda.get_device_name(0)}")
else:
    print("❌ PyTorch未启用CUDA，或显卡驱动未装")