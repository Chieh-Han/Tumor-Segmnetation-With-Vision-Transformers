import torch
import monai
from monai.networks.nets import SwinUNETR
import nibabel as nib

# 1. Check Versions
print(f"MONAI version: {monai.__version__}")
print(f"PyTorch version: {torch.__version__}")

# 2. Check GPU Access (Critical for your AMD card)
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")

if gpu_available:
    print(f"Using Device: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: GPU not detected. MONAI will run on CPU (very slow).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = nib.load("Task01_BrainTumour/imagesTr/BRATS_001.nii.gz")