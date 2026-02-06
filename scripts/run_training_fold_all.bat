@echo off
REM Train nnU-Net v2 on full dataset (fold_all)

REM Activate conda environment (edit this path!)
call "C:\Users\NukMed-AI\anaconda3\condabin\conda.bat" activate nnunet

REM Start training on all data
nnUNetv2_train Dataset001_SoftTissueDiana 2d all -tr nnUNetTrainer -p nnUNetPlans

pause
