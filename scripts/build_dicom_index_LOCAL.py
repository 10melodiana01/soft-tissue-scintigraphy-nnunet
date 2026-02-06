import argparse
from pathlib import Path
import pandas as pd
import pydicom

def get_tag(ds, name, default=""):
    return str(getattr(ds, name, default) or default)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sep", default=";")
    args = ap.parse_args()

    dicom_root = Path(args.dicom_root)
    rows = []

    for p in dicom_root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue

        rows.append({
            "file_path": str(p),
            "PatientID": get_tag(ds, "PatientID"),
            "PatientName": get_tag(ds, "PatientName"),
            "StudyInstanceUID": get_tag(ds, "StudyInstanceUID"),
            "SeriesInstanceUID": get_tag(ds, "SeriesInstanceUID"),
            "SOPInstanceUID": get_tag(ds, "SOPInstanceUID"),
            "StudyDate": get_tag(ds, "StudyDate"),
            "StudyTime": get_tag(ds, "StudyTime"),
            "AcquisitionDateTime": get_tag(ds, "AcquisitionDateTime"),
            "AccessionNumber": get_tag(ds, "AccessionNumber"),
            "StudyDescription": get_tag(ds, "StudyDescription"),
            "SeriesDescription": get_tag(ds, "SeriesDescription"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, sep=args.sep)
    print("Saved:", args.out_csv)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
