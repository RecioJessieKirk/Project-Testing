import torch
import torchvision

print(torch.__version__)  # Ensure this shows a CUDA version (e.g., 2.1.0+cu124)
print(torch.cuda.is_available())  # Should return True
print(torchvision.__version__)  # Ensure this is compatible with PyTorch
