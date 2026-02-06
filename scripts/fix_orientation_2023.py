import os
import numpy as np
import nibabel as nib

# INPUT: your current (wrongly oriented) nifti folder
IN_DIR = r"C:\Users\NukMed-AI\Desktop\Soft Tissue Diana\NIFTI_AUT2023"

# OUTPUT: new folder with fixed orientation (do NOT overwrite originals)
OUT_DIR = r"C:\Users\NukMed-AI\Desktop\Soft Tissue Diana\NIFTI_AUT2023_FIXED"

os.makedirs(OUT_DIR, exist_ok=True)

def rotate_180_in_plane(arr):
    # Works for shapes like (H, W), (H, W, F), (H, W, F, ...)
    # 180° rotation = flip both axes 0 and 1
    return np.flip(np.flip(arr, axis=0), axis=1)

for fn in os.listdir(IN_DIR):
    if not (fn.endswith(".nii") or fn.endswith(".nii.gz")):
        continue

    in_path = os.path.join(IN_DIR, fn)
    out_path = os.path.join(OUT_DIR, fn)

    img = nib.load(in_path)
    data = img.get_fdata()

    data_fixed = rotate_180_in_plane(data)

    # Keep the same affine/header to preserve spacing etc.
    out_img = nib.Nifti1Image(data_fixed, img.affine, img.header)
    nib.save(out_img, out_path)

    print("fixed:", fn)

print("\nDone.")
print("Fixed files written to:", OUT_DIR)
