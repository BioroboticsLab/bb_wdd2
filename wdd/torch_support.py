
cuda_is_available = False
torch = None

try:
    import torch
    cuda_is_available = torch.cuda.is_available()
except Exception:
    pass

if cuda_is_available:
    print("Torch/CUDA is available and will be used to speed up computations.")
else:
    torch = None
