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

# ── 1. Load file paths from dataset.json ─────────────────────────────────
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