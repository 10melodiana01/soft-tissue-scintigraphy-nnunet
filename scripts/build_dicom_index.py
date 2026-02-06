import argparse
import os
from pathlib import Path

import pandas as pd

# pydicom is the standard library to read DICOM headers
# install locally (NOT in repo): pip install pydicom
import pydicom


DICOM_TAGS = {
    "PatientID": ("0010", "0020"),
    "StudyInstanceUID": ("0020", "000D"),
    "SeriesInstanceUID": ("0020", "000E"),
    "StudyDate": ("0008", "0020"),
    "StudyTime": ("0008", "0030"),
    "Modality": ("0008", "0060"),
    "SeriesDescription": ("0008", "103E"),
    "ProtocolName": ("0018", "1030"),
}


def get_tag(ds, name, default=None):
    # safer: ds.get("PatientID", default) usually works, but we keep a robust version
    try:
        return str(getattr(ds, name))
    except Exception:
        return default


def iter_dicom_files(root: Path):
    # DICOM files may have arbitrary extensions; your screenshots show .dcm, but we do both:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            # quick filter (keeps it fast); extend if needed
            if p.suffix.lower() in {".dcm", ""}:
                yield p
            elif p.suffix.lower() in {".ima"}:
                yield p
            # else ignore


def build_index(roots, max_files_per_series=1):
    """
    We create one row per SeriesInstanceUID (series).
    We only read up to max_files_per_series files per series (fast), but also count total files.
    """
    series_rows = {}
    series_counts = {}

    for root in roots:
        for f in iter_dicom_files(root):
            # Try reading header only (fast). If it fails, skip.
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            except Exception:
                continue

            series_uid = get_tag(ds, "SeriesInstanceUID", None)
            if series_uid is None:
                continue

            # count files per series
            series_counts[series_uid] = series_counts.get(series_uid, 0) + 1

            # store one representative header per series
            if series_uid not in series_rows:
                series_rows[series_uid] = {
                    "dicom_root": str(root),
                    "dicom_example_file": str(f),
                    "patient_id": get_tag(ds, "PatientID", ""),
                    "study_uid": get_tag(ds, "StudyInstanceUID", ""),
                    "series_uid": series_uid,
                    "study_date": get_tag(ds, "StudyDate", ""),
                    "study_time": get_tag(ds, "StudyTime", ""),
                    "modality": get_tag(ds, "Modality", ""),
                    "series_description": get_tag(ds, "SeriesDescription", ""),
                    "protocol_name": get_tag(ds, "ProtocolName", ""),
                }

    # add counts
    for suid, row in series_rows.items():
        row["n_files_in_series"] = series_counts.get(suid, 0)

    df = pd.DataFrame(list(series_rows.values()))
    # sort for sanity
    sort_cols = [c for c in ["patient_id", "study_date", "study_time", "modality"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build a DICOM index table (one row per series) from raw DICOM folders.")
    parser.add_argument("--dicom_root", type=str, required=True,
                        help="Folder that contains DICOM subfolders (e.g. ...\\Data).")
    parser.add_argument("--subfolders", type=str, nargs="*", default=["DICOMS_AUT2020", "DICOMS_AUT2023"],
                        help="Which subfolders inside dicom_root to scan.")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path.")
    args = parser.parse_args()

    dicom_root = Path(args.dicom_root)
    roots = [dicom_root / s for s in args.subfolders]
    roots = [r for r in roots if r.exists()]
    if not roots:
        raise FileNotFoundError(f"No valid DICOM subfolders found under {dicom_root} (checked {args.subfolders})")

    df = build_index(roots)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, sep=";")
    print("Saved:", out_csv)
    print("Rows (series):", len(df))
    print("Unique patients:", df["patient_id"].nunique() if "patient_id" in df.columns else "n/a")


if __name__ == "__main__":
    main()
