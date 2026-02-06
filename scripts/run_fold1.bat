@echo off

call "C:\Users\NukMed-AI\anaconda3\condabin\conda.bat" activate nnunet

nnUNetv2_train Dataset001_SoftTissueDiana 2d 1 -tr nnUNetTrainer -p nnUNetPlans

pause
