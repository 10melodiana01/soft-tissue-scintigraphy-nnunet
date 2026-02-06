@echo off
setlocal

REM Activate conda env
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" nnunet

REM Train fold_all (single model trained on all training data)
nnUNetv2_train Dataset001_SoftTissueDiana 2d nnUNetTrainer__nnUNetPlans__2d fold_all

pause
endlocal
