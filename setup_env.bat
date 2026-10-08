@echo off
rem Conda Environment setup script for Windows (Intel PC + NVIDIA CUDA)

echo ==================================================
echo   Setting up CR_NNGP Conda Environment (cr_nngp)
echo ==================================================

rem Check if environment exists
call conda env list | findstr /b /c:"cr_nngp " >nul
if %errorlevel% equ 0 (
    echo Conda environment 'cr_nngp' already exists. Updating...
    call conda env update -f environment.yml --prune
) else (
    echo Creating Conda environment 'cr_nngp' from environment.yml...
    call conda env create -f environment.yml
)

rem Activate conda environment
echo Activating 'cr_nngp'...
call conda activate cr_nngp

rem Register Jupyter kernel
echo Registering Jupyter Kernel 'cr_nngp'...
python -m ipykernel install --user --name=cr_nngp --display-name "Python (cr_nngp)"

echo ==================================================
echo   Setup Complete! To activate the Conda environment:
echo     conda activate cr_nngp
echo ==================================================
