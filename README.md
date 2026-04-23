# Soybean Insect Resistance Screening from UAV RGB Images

This repository provides the code used for soybean genotype screening under contrasting pest-management conditions (Controlled vs NoControl), using UAV RGB imagery and dual feature streams (VI and DINOv3).

This release is prepared for journal submission and public reproducibility.

## Author and Affiliation

- Current maintainer: PhD student, ICMC, University of Sao Paulo (USP)
- Additional contributor names and contact details can be added later.

## Study Pipeline Summary

The full workflow includes both external preprocessing and in-repo analysis:

1. External data preparation (outside this repository):
   - UAV RGB acquisition across six time points.
   - Orthomosaic reconstruction in WebODM.
   - RTK/GCP-based co-registration and plot polygon delineation in QGIS.
2. In-repo data preparation:
   - Convert orthomosaic + shapefile + phenotype table into cropped plot images and `dataset_metadata.json`.
3. Parallel feature extraction:
   - Vegetation Index (VI) statistical features.
   - DINOv3 ViT-S/16 deep embeddings.
4. Analysis and ranking:
   - Cross-field similarity (Controlled vs NoControl), multi-view 3D scoring, and Z8 occupancy-based candidate ranking.

More detail is provided in:
- `docs/METHODS_OVERVIEW.md`
- `docs/REPRODUCIBILITY.md`
- `docs/DATA_LAYOUT.md`

## Main Programs

The project centers on three main scripts:

1. `experiments/image_and_label_preparation.py`
   - Builds cropped plot images and labels metadata (`dataset_metadata.json`) from externally prepared orthomosaic/shapefile inputs.
2. `experiments/extract_vi_dinov3.py`
   - Extracts VI and DINOv3 features.
3. `experiments/run_insect_resistance_analysis.py`
   - Runs the main insect resistance analysis and visualization workflow.

## Environment

Recommended environment:

```powershell
conda activate soy
pip install -r requirements.txt
```

## Configuration

The single active runtime config entry is:

- `src/config/config.py`

Set optional environment variables (PowerShell):

```powershell
$env:SOY_DATA_ROOT = "E:/YourData/AnhumasPiracicaba"
$env:SOY_CHECKPOINT_ROOT = "E:/YourData/checkpoints"
$env:SOY_OUTPUT_ROOT = "E:/YourData/soy_outputs"
$env:SOY_DEVICE = "cuda"
```

Notes:
- `SOY_DATA_ROOT` should point to a folder containing `dataset/`.
- If `SOY_DATA_ROOT` is not set, the default is `<repo_root>/AnhumasPiracicaba`.

## Run

Step-by-step:

```powershell
python .\experiments\image_and_label_preparation.py
python .\experiments\extract_vi_dinov3.py
python .\experiments\run_insect_resistance_analysis.py
```

One-command workflow:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_main_workflow.ps1
```

Skip data preparation if already completed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_main_workflow.ps1 -SkipDataPreparation
```

## Outputs

- Feature files: `outputs/features/` (or `SOY_OUTPUT_ROOT/features/` if set)
- Analysis results: `experiments/insect_resistance/outputs/results/`

## Reproducibility Scope

This repository assumes that orthomosaic generation and QGIS plot delineation are completed externally.
The code here starts from those prepared inputs and performs dataset assembly, feature extraction, and analysis.

## Data Availability

- This repository is designed to be reproduced from externally prepared geospatial assets.
- For public release, we recommend sharing processed data that directly supports reproducibility:
   - orthomosaics (per flight date),
   - plot boundary files (shapefile/geopackage),
   - cropped plot image dataset,
   - `dataset_metadata.json` and required trait tables.
- Due to size and institutional constraints, full raw UAV image sets may be provided on reasonable request instead of being mirrored in this Git repository.
- The expected local folder layout is documented in `docs/DATA_LAYOUT.md`.

## Code Availability

- Code is publicly available in this repository.
- Versioned citation metadata is provided in `CITATION.cff`.

## Citation

If you use this repository, please cite it using the metadata in `CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE`.
