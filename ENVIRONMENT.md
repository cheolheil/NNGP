# CR_NNGP Conda Environment & VS Code Setup Guide

This project is configured to run seamlessly as a **Conda Environment (`cr_nngp`)** across both **macOS (Apple Silicon M1/M2/M3/M4)** and **Windows (Intel + NVIDIA CUDA GPU)**, with native support in **VS Code**.

---

## 📦 Required Packages in Conda (`environment.yml`)
- `python>=3.10`
- `pytorch>=2.0.0`
- `numpy>=1.22.0`
- `scipy>=1.8.0`
- `matplotlib>=3.5.0`
- `ipykernel>=6.0.0`
- `notebook>=6.4.0`

---

## 🚀 Step 1: Create / Update the Conda Environment

### On macOS / Linux:
Open Terminal in this project folder and run:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### On Windows:
Open Command Prompt or PowerShell in this project folder and run:
```cmd
setup_env.bat
```

*Or manually using Conda directly:*
```bash
conda env create -f environment.yml
conda activate cr_nngp
python -m ipykernel install --user --name=cr_nngp --display-name "Python (cr_nngp)"
```

---

## 💻 Step 2: Configure VS Code for Seamless Execution

### A. Select the Python Interpreter for `.py` Scripts
1. Open **VS Code**.
2. Open the Command Palette:
   - **macOS**: `Cmd` + `Shift` + `P`
   - **Windows**: `Ctrl` + `Shift` + `P`
3. Type and select **`Python: Select Interpreter`**.
4. Choose **`cr_nngp (Conda)`** (or `Python 3.1x ('cr_nngp': conda)`).
5. Now, pressing the ▶️ **Play button** or `Shift + Enter` will automatically execute scripts using your `cr_nngp` Conda environment!

---

### B. Select the Jupyter Kernel for `.ipynb` Notebooks
1. Open any notebook (`notebook.ipynb` or `jax_test.ipynb`) in VS Code.
2. In the top-right corner of the notebook editor, click **Select Kernel**.
3. Select **Python Environments...** and choose **`Python (cr_nngp)`**.
