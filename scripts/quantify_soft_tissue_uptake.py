import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd


# Default label meaning (edit if needed)
DEFAULT_LABEL_MEANING = {
    0: "Background",
    1: "OS_soft",
    2: "OS_bone",
    3: "US_soft",
    4: "US_bone",
}

DEFAULT_MAPPING = {
    "OS_soft": 1,
    "OS_bone": 2,
    "US_soft": 3,
    "US_bone": 4,
}


def load_2d_any(path: Path, channel: int = 0, slice_idx=None) -> np.ndarray:
    """Load NIfTI and return a 2D array.
    Supports (H,W), (H,W,C), (H,W,Z), (H,W,Z,C).
    """
    arr = nib.load(str(path)).get_fdata()

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # (H,W,C) or (H,W,Z)
        if arr.shape[2] <= 4:
            if channel >= arr.shape[2]:
                raise ValueError(f"Channel {channel} out of range for {path} with shape {arr.shape}")
            return arr[:, :, channel]
        else:
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

    raise ValueError(f"Unsupported NIfTI shape {arr.shape} for {path}")


def safe_mean(x: np.ndarray):
    return float(x.mean()) if x.size else np.nan


def safe_ratio(num, den):
    if np.isnan(num) or np.isnan(den) or den == 0:
        return np.nan
    return float(num / den)


def quantify_case(image_path: Path, seg_path: Path, channel: int, mapping: dict) -> dict:
    img = load_2d_any(image_path, channel=channel)
    seg = load_2d_any(seg_path, channel=0)
    seg = np.rint(seg).astype(np.int32)

    if img.shape != seg.shape:
        raise ValueError(f"Shape mismatch: {image_path.name} img {img.shape} vs seg {seg.shape}")

    case_id = image_path.name.replace("_0000.nii.gz", "")

    os_soft = img[seg == mapping["OS_soft"]]
    os_bone = img[seg == mapping["OS_bone"]]
    us_soft = img[seg == mapping["US_soft"]]
    us_bone = img[seg == mapping["US_bone"]]

    os_soft_mean = safe_mean(os_soft)
    os_bone_mean = safe_mean(os_bone)
    us_soft_mean = safe_mean(us_soft)
    us_bone_mean = safe_mean(us_bone)

    return {
        "case_id": case_id,
        "OS_soft_mean": os_soft_mean,
        "OS_bone_mean": os_bone_mean,
        "OS_soft_to_bone_ratio": safe_ratio(os_soft_mean, os_bone_mean),
        "US_soft_mean": us_soft_mean,
        "US_bone_mean": us_bone_mean,
        "US_soft_to_bone_ratio": safe_ratio(us_soft_mean, us_bone_mean),
        "OS_soft_n": int(os_soft.size),
        "OS_bone_n": int(os_bone.size),
        "US_soft_n": int(us_soft.size),
        "US_bone_n": int(us_bone.size),
    }


def main():
    ap = argparse.ArgumentParser(description="Quantify soft-tissue uptake from nnU-Net segmentations (planar scintigraphy).")
    ap.add_argument("--images_dir", required=True, help="Folder with nnU-Net style inputs: <case>_0000.nii.gz")
    ap.add_argument("--segs_dir", required=True, help="Folder with predicted segmentations: <case>.nii.gz")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--channel", type=int, default=0, help="Image channel for (H,W,2) anterior/posterior. Default 0.")
    ap.add_argument("--sep", default=";", help="CSV separator, default ';'")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    segs_dir = Path(args.segs_dir)
    out_csv = Path(args.out_csv)

    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not segs_dir.exists():
        raise FileNotFoundError(segs_dir)

    image_files = sorted(images_dir.glob("*_0000.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No *_0000.nii.gz found in {images_dir}")

    rows = []
    missing = 0

    for img_path in image_files:
        case_id = img_path.name.replace("_0000.nii.gz", "")
        seg_path = segs_dir / f"{case_id}.nii.gz"

        if not seg_path.exists():
            missing += 1
            continue

        rows.append(quantify_case(img_path, seg_path, channel=args.channel, mapping=DEFAULT_MAPPING))

    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, sep=args.sep)

    print("Saved:", out_csv)
    print("Cases quantified:", len(df))
    print("Missing segmentations:", missing)


if __name__ == "__main__":
    main()
