import json
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

import monai
import torch
from monai.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
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
from monai.losses import DiceCELoss

# Constants

DATA_ROOT    = Path("Task01_BrainTumour")
WEIGHTS_URL  = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
WEIGHTS_PATH = Path("model_swinvit.pt")
CROP_SIZE    = (96, 96, 96)

#monai.torch.backends.cudnn.benchmark = False
#monai.torch.backends.cudnn.enabled   = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler("cuda")
# Data

def load_data():
    with open(DATA_ROOT / "dataset.json") as f:
        meta = json.load(f)

    all_cases = [
        {
            "image": str(DATA_ROOT / entry["image"].lstrip("./")),
            "label": str(DATA_ROOT / entry["label"].lstrip("./")),
        }
        for entry in meta["training"]
    ]

    split      = int(len(all_cases) * 0.8)
    train_list = all_cases[:split]
    val_list   = all_cases[split:]

    return train_list, val_list


def build_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=CROP_SIZE, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])

    return train_transforms, val_transforms


def build_loaders(train_list, val_list, train_transforms, val_transforms):
    train_ds = Dataset(data=train_list, transform=train_transforms)
    val_ds   = Dataset(data=val_list,   transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader

# Model

def build_model():
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    if not WEIGHTS_PATH.exists():
        print("Downloading pretrained weights...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)

    weight = torch.load(WEIGHTS_PATH)
    model.load_state_dict(weight["state_dict"], strict=False)
    print("Encoder weights loaded.")

    return model

# Sanity Checks

def check_system():
    print(f"MONAI version:  {monai.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: no GPU detected, will be very slow.")


def check_dataloader(train_loader):
    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}")  # (1, 4, 96, 96, 96)
    print(f"Label shape: {batch['label'].shape}")  # (1, 3, 96, 96, 96)
    print(f"Image dtype: {batch['image'].dtype}")
    print(f"Label dtype: {batch['label'].dtype}")


def check_model(model):
    with torch.no_grad():
        dummy = torch.randn(1, 4, 96, 96, 96).to(device)
        out   = model(dummy)
        print(f"Model output shape: {out.shape}")  # (1, 3, 96, 96, 96)

    print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"VRAM reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    torch.cuda.empty_cache()
    
    
# Train    

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training", ncols=80):
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"): # autocase casts forward pass into float16 calculation to save time
            output = model(image)
            loss   = loss_fn(output, label)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        total_loss += loss.item()
    return total_loss / len(loader)


# Main

if __name__ == "__main__":
    check_system()

    train_list, val_list             = load_data()
    train_transforms, val_transforms = build_transforms()
    train_loader, val_loader         = build_loaders(train_list, val_list, train_transforms, val_transforms)

    print(f"Train: {len(train_list)} | Val: {len(val_list)}")

    # check_dataloader(train_loader)

    model = build_model()
    
    # check_model(model)

    # print("All checks passed. Ready to train.")

    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epoch = 3
    start_epoch = 0
    
    best_loss = float("inf")
    train_losses = []
    
    if Path("checkpoint_latest.pt").exists():
        checkpoint = torch.load("checkpoint_latest.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting from scratch.")

    for epoch in range(start_epoch, num_epoch):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        train_losses.append(loss) # records losses
        print(f"Epoch {epoch+1}/{num_epoch} | Loss: {loss:.4f}")

        # always save latest for resuming
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "train_losses": train_losses,
            "best_loss": best_loss,
        }, "checkpoint_latest.pt")

        # only save best when loss improves
        if loss < best_loss:
            best_loss = loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, "checkpoint_best.pt")
            print(f"New best saved: {best_loss:.4f}")