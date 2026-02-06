# soft-tissue-scintigraphy-nnunet
Deep learning–based soft tissue segmentation and quantitative analysis in planar bone scintigraphy for cardiac amyloidosis assessment.

# Soft Tissue Segmentation & Quantification in Bone Scintigraphy

This repository contains the code and analysis pipeline developed for a research project on 
**deep learning–based soft tissue segmentation in planar bone scintigraphy**, with the aim of 
enabling **standardized and reproducible quantification of soft tissue uptake** related to 
systemic amyloid deposition in patients with suspected cardiac amyloidosis.

## Background
Cardiac amyloidosis is a progressive disease characterized by the deposition of misfolded amyloid 
proteins in the myocardium, leading to restrictive cardiomyopathy and heart failure. 
Planar bone scintigraphy is widely used in nuclear medicine and has demonstrated high diagnostic 
value, but current evaluation approaches are largely qualitative or semi-quantitative and rely on 
visual assessment, resulting in inter-observer and inter-center variability.

This project investigates whether **deep learning–based segmentation** can reliably delineate 
soft tissue regions in planar bone scintigraphy images and serve as a foundation for 
**robust quantitative imaging biomarkers**.

## Project Scope
- nnU-Net v2–based segmentation of soft tissue and bone regions in planar bone scintigraphy
- Quantitative pixel-based analysis of segmented regions
- Extraction of soft tissue uptake metrics normalized to bone reference regions
- Cross-validation and performance evaluation using standard segmentation metrics

## Repository Contents
- `src/` – Core Python code for quantitative feature extraction and analysis  
- `scripts/` – Helper scripts and execution wrappers  
- `notebooks/` – Jupyter notebooks for analysis and visualization (without data)  
- `configs/` – Configuration templates (paths, parameters)  
- `docs/` – Project documentation and methodological notes  

## Data Availability
⚠️ **No imaging data, labels, or model checkpoints are included in this repository.**  
Due to data protection and ethical constraints, all patient data, predictions, and trained models 
are excluded. The repository contains **code only**.

## Requirements
- Python ≥ 3.9  
- nnU-Net v2  
- NumPy, pandas, nibabel, matplotlib  

## Disclaimer
This repository is intended for research and educational purposes only and is **not approved for 
clinical use**.

## Citation
If you use or build upon this work, please cite:

Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*,  
Nature Methods, 2021.
