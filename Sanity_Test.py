import torch
import monai
from monai.networks.nets import SwinUNETR
import nibabel as nib
import numpy as np
import sys; 

### -------- Check Hardware Compatibility --------
print(f"MONAI version: {monai.__version__}")
print(f"PyTorch version: {torch.__version__}")

gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")

if gpu_available:
    print(f"Using Device: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: GPU not detected. MONAI will run on CPU (very slow).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load Data

img = nib.load("Task01_BrainTumour/imagesTr/BRATS_002.nii.gz")
data = img.get_fdata(dtype=np.float32)  # (240, 240, 155, 4)

### -------- Data Pre-Processing --------

data = np.transpose(data, (3, 0, 1, 2))  # (4, H, W, D)
print(f"After transpose: {data.shape}")

_, H, W, D = data.shape

# crop center 128 cube for testing
PATCH = 128

cx = H // 2
cy = W // 2
cz = D // 2

# Check the volume is actually large enough to crop from
assert H >= PATCH, f"Volume too small in X: {H}"
assert W >= PATCH, f"Volume too small in Y: {W}"
assert D >= PATCH, f"Volume too small in Z: {D}"

patch = data[:,
             cx - 64 : cx + 64,
             cy - 64 : cy + 64,
             cz - 64 : cz + 64]

print(f"Patch shape: {patch.shape}")  # should be (4, 128, 128, 128)

# tranform to torch tensor

tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
# put numpy into torch tensor
# add batch dimention to tensor (1, 4, 128, 128, 128)
# add tensor to GPU VRAM

print(f"Input tensor shape: {tensor.shape}")   # expect (1, 4, 128, 128, 128)
print(f"Input tensor device: {tensor.device}") # expect cuda

### -------- Build Model --------

model = SwinUNETR(
    in_channels=4,    # FLAIR, T1, T1gd, T2
    out_channels=3,   # NCR, ED, ET
    feature_size=48,
    use_checkpoint=True,  # gradient checkpointing — saves VRAM
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

### -------- Model Sanity Test --------
model.eval()
with torch.no_grad():
    output = model(tensor)

print(f"Output shape: {output.shape}")  # expect (1, 3, 128, 128, 128)
print("Forward pass successful!")

### -------- Check VRAM Usage --------
print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"VRAM reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")

# free GPU memory and exit
torch.cuda.empty_cache()
print("Done.")
sys.exit(0)