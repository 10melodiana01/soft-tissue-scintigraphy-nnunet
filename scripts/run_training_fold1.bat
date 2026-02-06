@echo off
REM Train nnU-Net v2 fold 1 (2D)

REM Activate conda environment (edit this path!)
call "C:\Users\NukMed-AI\anaconda3\condabin\conda.bat" activate nnunet

REM Start training
nnUNetv2_train Dataset001_SoftTissueDiana 2d 1 -tr nnUNetTrainer -p nnUNetPlans

pause
