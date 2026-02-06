import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd


# Label mapping (verified via overlay inspection)
# 1 = OS soft tissue, 2 = OS bone, 3 = US soft tissue, 4 = US bone
MAPPING = {
    "OS_knochen": 2,
    "OS_weich":   1,
    "US_knochen": 4,
    "US_weich":   3,
}


def load_2d_any(path: Path, slice_idx=None, channel: int = 0) -> np.ndarray:
    """Load NIfTI and return a 2D array.
    Supports shapes: (H,W), (H,W,C), (H,W,Z), (H,W,Z,C).
    """
    arr = nib.load(str(path)).get_fdata()

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # (H,W,C) or (H,W,Z)
        if arr.shape[2] <= 4:  # likely channels/frames
            if channel >= arr.shape[2]:
                raise ValueError(f"Channel {channel} out of range for {path} with shape {arr.shape}")
            return arr[:, :, channel]
        # likely slices
        if slice_idx is None:
            slice_idx = arr.shape[2] // 2
        return arr[:, :, slice_idx]

    if arr.ndim == 4:
        # (H,W,Z,C)
        if slice_idx is None:
            slice_idx = arr.shape[2] // 2
        if channel >= arr.shape[3]:
            raise ValueError(f"Channel {channel} out of range for {path} with shape {arr.shape}")
        return arr[:, :, slice_idx, channel]

    raise ValueError(f"Unsupported shape {arr.shape} in {path}")


def safe_mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else np.nan


def safe_ratio(num: float, den: float) -> float:
    if np.isnan(num) or np.isnan(den) or den == 0:
        return np.nan
    return float(num / den)


def compute_means_and_ratios(img_path: Path, label_path: Path, channel: int = 0) -> dict:
    img = load_2d_any(img_path, channel=channel)
    lab = load_2d_any(label_path, channel=0)
    lab = np.rint(lab).astype(np.int32)

    if img.shape != lab.shape:
        raise ValueError(f"Shape mismatch: image {img.shape} vs label {lab.shape} for {img_path.name}")

    case_id = img_path.name.replace("_0000.nii.gz", "")

    os_bone = img[lab == MAPPING["OS_knochen"]]
    os_soft = img[lab == MAPPING["OS_weich"]]
    us_bone = img[lab == MAPPING["US_knochen"]]
    us_soft = img[lab == MAPPING["US_weich"]]

    os_bone_mean = safe_mean(os_bone)
    os_soft_mean = safe_mean(os_soft)
    us_bone_mean = safe_mean(us_bone)
    us_soft_mean = safe_mean(us_soft)

    return {
        "case_id": case_id,
        "OS_knochen_mean": os_bone_mean,
        "OS_weich_mean": os_soft_mean,
        "US_knochen_mean": us_bone_mean,
        "US_weich_mean": us_soft_mean,
        "OS_weich_to_knochen_ratio": safe_ratio(os_soft_mean, os_bone_mean),
        "US_weich_to_knochen_ratio": safe_ratio(us_soft_mean, us_bone_mean),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean uptake and soft-to-bone ratios from planar scintigraphy images using segmentation masks."
    )
    parser.add_argument("--images_dir", type=str, required=True, help="Folder with nnU-Net images (e.g. *_0000.nii.gz).")
    parser.add_argument("--labels_dir", type=str, required=True, help="Folder with label masks (e.g. <case_id>.nii.gz).")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--channel", type=int, default=0, help="Channel index (0=anterior/front).")

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_csv = Path(args.out_csv)

    image_files = sorted(images_dir.glob("*_0000.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No *_0000.nii.gz files found in {images_dir}")

    rows = []
    for img_path in image_files:
        case_id = img_path.name.replace("_0000.nii.gz", "")
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            continue

        rows.append(compute_means_and_ratios(img_path, label_path, channel=args.channel))

    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, sep=";")
    print("Saved:", str(out_csv))
    print("N cases:", len(df))


if __name__ == "__main__":
    main()
