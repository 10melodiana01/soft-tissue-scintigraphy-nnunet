import nibabel as nib
import numpy as np
from pathlib import Path

# Input folders (already fixed/orientation-corrected NIfTIs)
SRC_2020 = Path(r"PATH")
SRC_2023 = Path(r"PATH")

# Output folder for nnU-Net inference (posterior view)
OUT = Path(r"PATH")
OUT.mkdir(exist_ok=True)

# "Clean header" affine (identity)
I = np.eye(4, dtype=np.float32)

# If (H,W,2): define which index is posterior
POST_INDEX = 1  # <-- change to 0 if your dataset stores POST at index 0

def extract_post_as_hw1(x):
    """returns (H,W,1) posterior or None if unsupported"""
    if x.ndim == 2:
        # single-frame images: no posterior available
        return None
    if x.ndim == 3 and x.shape[2] == 1:
        # single-frame images: no posterior available
        return None
    if x.ndim == 3 and x.shape[2] == 2:
        return x[:, :, POST_INDEX][:, :, None]
    return None  # skip 4D etc.

def process(folder: Path, prefix: str):
    files = sorted(folder.glob("*.nii*"))
    written = 0
    skipped = 0

    for i, p in enumerate(files, start=1):
        nii = nib.load(str(p))
        x = np.asanyarray(nii.dataobj).astype(np.float32)

        post = extract_post_as_hw1(x)
        if post is None:
            skipped += 1
            continue

        # cleanheader: write with identity affine + qform/sform
        out_nii = nib.Nifti1Image(post, I)
        out_nii.set_qform(I, code=1)
        out_nii.set_sform(I, code=1)

        out_path = OUT / f"{prefix}_{i:06d}_0000.nii.gz"
        nib.save(out_nii, str(out_path))
        written += 1

        if written % 2000 == 0:
            print(prefix, "written:", written, "skipped:", skipped)

    print(prefix, "DONE | written:", written, "skipped:", skipped)
    return written, skipped

w1 = process(SRC_2020, "AUT2020")
w2 = process(SRC_2023, "AUT2023")

print("TOTAL written:", w1[0] + w2[0])
print("TOTAL skipped:", w1[1] + w2[1])
print("OUT count:", len(list(OUT.glob("*.nii*"))))
print("OUT:", OUT)
