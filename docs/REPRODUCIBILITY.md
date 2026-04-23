# Reproducibility Guide

## 1) Environment

Recommended baseline:

- OS: Windows 10/11 or Linux
- Python: 3.10+
- Environment: conda env named soy (recommended)

Install dependencies:

```powershell
conda activate soy
pip install -r requirements.txt
```

## 2) Configure Paths

Set environment variables if data/checkpoints are stored outside repository:

```powershell
$env:SOY_DATA_ROOT = "E:/YourData/AnhumasPiracicaba"
$env:SOY_CHECKPOINT_ROOT = "E:/YourData/checkpoints"
$env:SOY_OUTPUT_ROOT = "E:/YourData/soy_outputs"
$env:SOY_DEVICE = "cuda"
```

Runtime config file: src/config/config.py

## 3) Required Input Assets

External preprocessing must be done before running this repository:

1. Orthomosaic reconstruction from UAV RGB images (WebODM).
2. Plot polygon delineation and geospatial alignment (QGIS + RTK/GCP support).
3. Trait table (for labels) in CSV/Excel format.

Expected structure is documented in docs/DATA_LAYOUT.md.

## 4) Execution Order

Option A: step-by-step

```powershell
python .\experiments\image_and_label_preparation.py
python .\experiments\extract_vi_dinov3.py
python .\experiments\run_insect_resistance_analysis.py
```

Option B: one command

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_main_workflow.ps1
```

Skip data preparation if already done:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_main_workflow.ps1 -SkipDataPreparation
```

## 5) Expected Artifacts

- Feature outputs:
  - outputs/features/dinov3/*.pkl
  - outputs/features/vegetation_indices/*.pkl
  - or SOY_OUTPUT_ROOT/features/... when output root is overridden
- Analysis outputs:
  - experiments/insect_resistance/outputs/results/...

## 6) Minimal Verification

Run:

```powershell
python .\experiments\extract_vi_dinov3.py
```

Successful execution with generated PKL files indicates path and dependency setup is correct.

## 7) Versioning Notes

For manuscript submission, tag a release corresponding to the analyzed results and cite that version in the paper.
