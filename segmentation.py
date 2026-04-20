import json
import os
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    ConvertToMultiChannelBasedOnBratsClassesd,
    ToTensord,
)
from monai.data import Dataset, DataLoader

# Load Data 
data_root = Path("Task01_BrainTumour")

with open(data_root / "dataset.json") as f:
    meta = json.load(f)

# build list of dicts with absolute paths
all_cases = [
    {
        "image": str(data_root / entry["image"].lstrip("./")),
        "label": str(data_root / entry["label"].lstrip("./")),
    }
    for entry in meta["training"]
]

# split 80/20 train/val
split      = int(len(all_cases) * 0.8)
train_list = all_cases[:split]
val_list   = all_cases[split:]

print(f"Total cases:      {len(all_cases)}")
print(f"Training cases:   {len(train_list)}")
print(f"Validation cases: {len(val_list)}")

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]), # loads data into numpy arrays from dictionary of file path
    EnsureChannelFirstd(keys=["image", "label"]), # NIfTI store the channel dimension last, PyTorch expects first, switch to pytorch format

    # convert integer labels 0,1,2,3 to 3 binary channels
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    # Brats use 1,2,3 to label different part of the tumor, their addition makes up 
    # whole_tumor = (label == 1) | (label == 2) | (label == 3)  # all three
    # tumor_core  = (label == 1) | (label == 3)                  # skip edema
    # enhancing   = (label == 3)                                  # just ET
    
    
    # z-score normalize each modality over nonzero voxels
    NormalizeIntensityd(
        keys="image",
        nonzero=True,
        channel_wise=True
    ),

    # random 128³ crop — different every iteration
    RandSpatialCropd(
        keys=["image", "label"],
        roi_size=(128, 128, 128),
        random_size=False,
    ),

    # augmentation
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

    ToTensord(keys=["image", "label"]),
])

