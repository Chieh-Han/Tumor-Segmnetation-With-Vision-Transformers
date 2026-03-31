import torch
import monai
from monai.networks.nets import SwinUNETR

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

# Initialize for 1.5.2 (img_size is no longer a keyword)
model = SwinUNETR(
    spatial_dims=3,     # 3 for 3D volumes (MRI/CT)
    in_channels=4,      # e.g., 4 modalities for BraTS (T1, T1c, T2, FLAIR)
    out_channels=3,     # tumor regions (WT, TC, ET)
    feature_size=48,    # Standard for the high-performance Swin variant
    use_checkpoint=True # CRITICAL for your 16GB VRAM to avoid OOM
).to(device)

print("SwinUNETR 1.5.2 initialized successfully!")