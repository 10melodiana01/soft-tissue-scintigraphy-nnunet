import argparse
from pathlib import Path
import pandas as pd


def pick_col(df, candidates):
    """Return first column name that exists in df.columns from candidates list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_cols(df):
    """Strip whitespace from column names."""
    df.columns = [c.strip() for c in df.columns]
    return df


def main(dicom_index, nifti_index, out_csv, sep=";"):
    dicom_index = Path(dicom_index)
    nifti_index = Path(nifti_index)
    out_csv = Path(out_csv)

    df_d = pd.read_csv(dicom_index, sep=sep, dtype=str)
    df_n = pd.read_csv(nifti_index, sep=sep, dtype=str)

    df_d = normalize_cols(df_d)
    df_n = normalize_cols(df_n)

    # --- detect date/time columns ---
    d_date = pick_col(df_d, ["StudyDate", "study_date", "date", "DATE"])
    d_time = pick_col(df_d, ["StudyTime", "study_time", "time", "TIME"])

    n_date = pick_col(df_n, ["StudyDate", "study_date", "date", "DATE"])
    n_time = pick_col(df_n, ["StudyTime", "study_time", "time", "TIME"])

    if not d_date or not d_time:
        raise ValueError(f"DICOM index is missing date/time columns. Found: {list(df_d.columns)}")
    if not n_date or not n_time:
        raise ValueError(f"NIfTI index is missing date/time columns. Found: {list(df_n.columns)}")

    # --- normalize (trim spaces) ---
    df_d[d_date] = df_d[d_date].astype(str).str.strip()
    df_d[d_time] = df_d[d_time].astype(str).str.strip()
    df_n[n_date] = df_n[n_date].astype(str).str.strip()
    df_n[n_time] = df_n[n_time].astype(str).str.strip()

    # --- key ---
    df_d["key"] = df_d[d_date] + "_" + df_d[d_time]
    df_n["key"] = df_n[n_date] + "_" + df_n[n_time]

    # Optional: prefer ANT before POST if SeriesDescription exists
    if "SeriesDescription" in df_d.columns:
        df_d["SeriesDescription"] = df_d["SeriesDescription"].fillna("").astype(str)
        df_d = df_d.sort_values(["key", "SeriesDescription"])

    df_d_one = df_d.drop_duplicates(subset=["key"], keep="first").copy()

    keep_cols = [
        "key",
        "PatientID", "PatientName",
        "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
        "AccessionNumber",
        "StudyDescription", "SeriesDescription",
        "file_path",
        d_date, d_time,
    ]
    keep_cols = [c for c in keep_cols if c in df_d_one.columns]

    df_out = df_n.merge(df_d_one[keep_cols], on="key", how="left")

    # match flag
    df_out["match_found"] = df_out["PatientID"].notna().astype(int)

    # rename nifti date/time to standard output names
    df_out = df_out.rename(columns={n_date: "StudyDate", n_time: "StudyTime"})

    # reorder columns nicely (only if they exist)
    front = ["nifti_filename", "nifti_path", "StudyDate", "StudyTime", "match_found"]
    front = [c for c in front if c in df_out.columns]
    rest = [c for c in df_out.columns if c not in front]
    df_out = df_out[front + rest]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, sep=sep)

    n_total = len(df_out)
    n_match = int(df_out["match_found"].sum())
    print("Saved:", out_csv)
    print(f"Matched {n_match}/{n_total} NIfTIs ({(100*n_match/n_total):.1f}%)")

    if n_match < n_total:
        print("\nFirst unmatched rows:")
        cols_show = [c for c in ["nifti_filename", "StudyDate", "StudyTime"] if c in df_out.columns]
        print(df_out[df_out["match_found"] == 0][cols_show].head(10))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_index", required=True)
    ap.add_argument("--nifti_index", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sep", default=";")
    args = ap.parse_args()

    main(args.dicom_index, args.nifti_index, args.out_csv, sep=args.sep)
