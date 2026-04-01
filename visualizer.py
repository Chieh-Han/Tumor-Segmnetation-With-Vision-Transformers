import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

img = nib.load("Task01_BrainTumour/imagesTr/BRATS_001.nii.gz")

print(img.shape)        # (240, 240, 155, 4)
print(img.header.get_zooms())  # voxel size in mm, e.g. (1.0, 1.0, 1.0, 1.0)

data = img.get_fdata()
modalities = ["FLAIR", "T1", "T1gd", "T2"]

flair = data[..., 0]    # FLAIR (Fluid Attenuated Inversion Recovery)
t1    = data[..., 1]    # T1 
t1gd  = data[..., 2]    # T1 with gadolinium contrast — shows enhancing tumor
t2    = data[..., 3]    # T2

# Start in the middle slice
slice_idx = data.shape[2] // 2

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
plt.subplots_adjust(bottom=0.15, hspace=0.3)

ims = []
for i, ax in enumerate(axes.flat):
    vol = data[..., i]
    s   = vol[:, :, slice_idx]
    im  = ax.imshow(s, cmap="gray", origin="lower")
    ax.set_title(modalities[i], fontsize=12)
    ax.axis("off")
    ims.append((im, vol))

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider    = widgets.Slider(ax_slider, "Slice", 0, data.shape[2] - 1,
                           valinit=slice_idx, valstep=1)

def update(val):
    s = int(slider.val)
    for im, vol in ims:
        slc = vol[:, :, s]
        im.set_data(slc)
        im.set_clim(vmin=slc.min(), vmax=slc.max())
    fig.suptitle(f"Slice {s} / {data.shape[2]-1}", fontsize=11)
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()