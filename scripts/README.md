# Scripts

This folder contains helper scripts used for training, inference and quantitative analysis
in the soft tissue scintigraphy nnU-Net project.

The scripts are intended to document the processing pipeline and enable reproducibility.
No imaging data or patient-specific information is included or referenced directly.

## Overview

- `run_training_fold1.bat`  
  Starts nnU-Net training for a specific cross-validation fold.

- `run_training_fold_all.bat`  
  Trains the model on the full dataset (fold_all).

- `run_inference_fold.py`  
  Runs inference using a trained nnU-Net model to generate segmentation predictions.

- `quantify_soft_tissue_uptake.py`  
  Computes quantitative soft tissue uptake metrics based on segmentation masks and
  planar bone scintigraphy images.

Evaluation and visualization of segmentation metrics (e.g. Dice, IoU) are performed
in Jupyter notebooks located in the `notebooks/` directory.
