import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import monai
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
from monai.networks.nets import SwinUNETR
import urllib.request

# Fix ROCm bug with cuda
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

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

# Pre-Process Transforms for Training Data

train_transforms = Compose([ # allows chainning varias pre-processes
    LoadImaged(keys=["image", "label"]), # loads data into numpy arrays from dictionary of file path
    EnsureChannelFirstd(keys=["image", "label"]), # NIfTI store the channel dimension last, PyTorch expects first, switch to pytorch format

    # convert integer labels 0,1,2,3 to 3 binary channels
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),    # shape: (3, 240, 240, 155)
    
    # Brats use 1,2,3 to label different part of the tumor 
    # whole_tumor = (label == 1) | (label == 2) | (label == 3)  # all three
    # tumor_core  = (label == 1) | (label == 3)                 # skip edema
    # enhancing   = (label == 3)                                # just ET
    
    # z-score normalize each modality over nonzero voxels
    NormalizeIntensityd(
        keys="image",
        nonzero=True,
        channel_wise=True
    ), 

    # random 128³ crop — every iteration sees different cube but eventually sees the whole brain
    RandSpatialCropd(
        keys=["image", "label"], # need to crop both image and label
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

# Pre-Process Transforms for Validation Data

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"]),
    # no crop for validation — sliding window handles the full volume
])

# Dataset: What to load (Connection of training list w/ tranform, not actually linked till called upon)

train_ds = Dataset(data=train_list, transform=train_transforms)
val_ds   = Dataset(data=val_list,   transform=val_transforms)

# DataLoader: How to load it (Calls dataset during training to link training data w/ transforms)
# Acutally data is only loaded onto ram when calling in next(iter()) or for loop

train_loader = DataLoader(
    train_ds,
    batch_size=1,       # one volume per step, common for 3D medical imaging due to large data dim
    shuffle=True,       # randomize order each epoch
    num_workers=4,      # parallel data loading
    pin_memory=True,    # faster CPU→GPU transfer
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,      # always same order for validation
    num_workers=4,
)

# Dataloader sanity check
if __name__ == "__main__": # wrapper from guarding multiple worker from starting a fresh process
    print(f"Total cases:      {len(all_cases)}")
    print(f"Training cases:   {len(train_list)}")
    print(f"Validation cases: {len(val_list)}")

    batch = next(iter(train_loader)) # load the first sample in train_loader
    
    print(batch["image"].shape) # expect (1, 4, 128, 128, 128) (batch, channel, b, w, h)
    print(batch["label"].shape) # expect (1, 3, 128, 128, 128)
    print(batch["image"].dtype) # expect float32
    print(batch["label"].dtype) # expect bool since we switch it to 3 binary channels

    '''
    # Check how many samples
    for i, batch in enumerate(train_loader):
        pass
    print(f"number of training samples: {i}")

    # Quick look for the 1 sample
    img = batch["image"][0]  # (4, 128, 128, 128)
    lbl = batch["label"][0]  # (3, 128, 128, 128)
    s = 64  # middle slice

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    modalities = ["FLAIR", "T1", "T1gd", "T2"]
    for i in range(4):
        axes[0, i].imshow(img[i, :, :, s], cmap="gray")
        axes[0, i].set_title(modalities[i])

    labels = ["Whole Tumor", "Tumor Core", "Enhancing"]
    for i in range(3):
        axes[1, i].imshow(lbl[i, :, :, s], cmap="hot")
        axes[1, i].set_title(labels[i])

    axes[1, 3].axis("off")
    plt.tight_layout()
    plt.show()
    '''
    # Check Hardware Compatibility
    print(f"MONAI version: {monai.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")

    if gpu_available:
        print(f"Using Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: GPU not detected. MONAI will run on CPU (very slow).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")