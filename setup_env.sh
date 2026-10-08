#!/bin/bash
# Conda Environment setup script for macOS (Apple Silicon / Intel) & Linux

echo "=================================================="
echo "  Setting up CR_NNGP Conda Environment (cr_nngp)"
echo "=================================================="

# Initialize conda for shell if needed
eval "$(conda shell.bash hook 2>/dev/null)" || eval "$(conda shell.zsh hook 2>/dev/null)"

# Check if conda environment 'cr_nngp' already exists
if conda env list | grep -q "^cr_nngp "; then
    echo "Conda environment 'cr_nngp' already exists. Updating..."
    conda env update -f environment.yml --prune
else
    echo "Creating Conda environment 'cr_nngp' from environment.yml..."
    conda env create -f environment.yml
fi

# Activate conda environment
echo "Activating 'cr_nngp'..."
conda activate cr_nngp

# Register Jupyter kernel for VS Code / Jupyter
echo "Registering Jupyter Kernel 'cr_nngp'..."
python -m ipykernel install --user --name=cr_nngp --display-name "Python (cr_nngp)"

echo "=================================================="
echo "  Setup Complete! To activate the Conda environment:"
echo "    conda activate cr_nngp"
echo "=================================================="
