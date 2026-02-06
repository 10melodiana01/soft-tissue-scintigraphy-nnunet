import argparse
import csv
import re
from pathlib import Path

def extract_datetime(name):
    """
    Sucht YYYYMMDDHHMMSS im Dateinamen
    """
    m = re.search(r"(20\d{6})(\d{6})", name)
    if m:
        return m.group(1), m.group(2)
    return None, None

def main(nifti_root, out_csv, sep):
    nifti_root = Path(nifti_root)

    rows = []
    for f in nifti_root.glob("*.nii.gz"):
        date, time = extract_datetime(f.name)
        rows.append({
            "nifti_path": str(f),
            "nifti_filename": f.name,
            "StudyDate": date,
            "StudyTime": time
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys(), delimiter=sep)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--sep", default=";")
    args = parser.parse_args()

    main(args.nifti_root, args.out_csv, args.sep)
