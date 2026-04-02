import vedo
import nibabel as nib
import numpy as np

img   = nib.load("Task01_BrainTumour/imagesTr/BRATS_001.nii.gz")

label_img = nib.load("Task01_BrainTumour/labelsTr/BRATS_001.nii.gz")
labels    = label_img.get_fdata(dtype=np.float32)

data = img.get_fdata(dtype=np.float32)

print(labels.shape)  # (240, 240, 155)
print(np.unique(labels))  # [0. 1. 2. 3.] the 4 classes of label


flair = data[..., 0]    # T2 but with CSF suppressed, so periventricular lesions aren't washed out by fluid signal.
t1    = data[..., 1]    # Anatomy scan. Fat/white matter bright, water dark. Think "T1 = Tissue structure."
t1gd  = data[..., 2]    # T1 after gadolinium contrast. Bright spots = blood-brain barrier breakdown (tumors, active inflammation).
t2    = data[..., 3]    # Pathology detector. Water/CSF bright. Edema and lesions show up well.

# Volume object — vedo handles the meshing internally

brain = vedo.Volume(flair).isosurface()
brain.alpha(0.1).color('lightblue')

ncr   = vedo.Volume((labels == 1).astype(np.float32)).isosurface(0.5) # Non-Contrast-Enhancing Region : tumor or abnormal tissue that’s less active or infiltrative
edema = vedo.Volume((labels == 2).astype(np.float32)).isosurface(0.5) # Surrounding Swelling : not tumor itself, but reacts to it
et    = vedo.Volume((labels == 3).astype(np.float32)).isosurface(0.5) # Enhancing Tumor : active, aggressive tumor core

'''
Bright in T1gd → likely enhancing tumor
Bright in FLAIR/T2 → edema
Mixed/dark core → necrosis/NCR
'''

ncr.color('brown').alpha(0.8)
edema.color('blue').alpha(0.6)
et.color('red').alpha(0.9)

vedo.show(brain, ncr, edema, et,
          title="Brain tumor 3D",
          bg='black', 
          axes=0)