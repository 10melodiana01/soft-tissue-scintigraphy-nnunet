\# Scripts



This folder contains small helper scripts used in the project workflow.

\*\*No imaging data or patient-identifying information is stored in this repository.\*\*

All scripts are designed to run locally on secured data.



\## Overview



\### `build\_dicom\_index.py`

Scans a DICOM directory and exports a metadata index (`dicom\_index\_full.csv`).

The index is used to link imaging data to clinical records without relying on file order.



\*\*Outputs (example columns):\*\*

\- PatientID, PatientName

\- StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID

\- StudyDate, StudyTime, AcquisitionDateTime

\- AccessionNumber, Study/SeriesDescription

\- file\_path



\### `build\_nifti\_index.py`

Creates an index for converted NIfTI files by parsing acquisition timestamps from filenames.

Produces `nifti\_index.csv` with `StudyDate`/`StudyTime` extracted from the filename.



\### `build\_dicom\_nifti\_reference.py`

Builds a DICOM↔NIfTI reference table by joining `dicom\_index\_full.csv` and `nifti\_index.csv`

using `StudyDate + StudyTime` as a key.

Produces `dicom\_nifti\_reference.csv`.



\### `prepare\_nnunet\_inference\_inputs.py`

Copies NIfTI files into a new folder and renames them to nnU-Net compliant filenames:

`<case\_id>\_0000.nii.gz`.

Creates `nnunet\_inference\_mapping.csv` to map nnU-Net case IDs back to original filenames.



\### `merge\_reference\_tables.py`

Merges `dicom\_nifti\_reference.csv` with `nnunet\_inference\_mapping.csv` to create a master mapping:

`nnunet\_case\_id → original\_nifti → DICOM metadata`.



\### `quantify\_soft\_tissue\_uptake.py`

Quantifies regional uptake values using nnU-Net predicted segmentations.

Extracts pixel intensities per region (OS/US × soft/bone), computes mean uptake and

soft-to-bone ratios, and exports a CSV.



\## Notes

\- nnU-Net inference expects input filenames like `<case\_id>\_0000.nii.gz`.

\- Predicted segmentations are typically written as `<case\_id>.nii.gz` in the output folder.

\- Do \*\*not\*\* commit any DICOM/NIfTI data, labels, or clinical spreadsheets to this repository.



