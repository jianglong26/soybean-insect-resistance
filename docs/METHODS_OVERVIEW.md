# Methods Overview (Manuscript-Oriented Summary)

## Scope

This document summarizes the analytical methodology implemented by the public codebase.

The pipeline is end-to-end from externally prepared orthomosaics and shapefiles to candidate genotype ranking.

## End-to-End Workflow

1. UAV RGB acquisition over two paired fields:
   - Controlled (insecticide-protected)
   - NoControl (natural pest pressure)
2. Orthomosaic reconstruction in WebODM (external step).
3. Spatial co-registration and plot boundary delineation in QGIS using RTK/GCP references (external step).
4. Dataset assembly in this repository:
   - Plot cropping and metadata/label generation.
5. Parallel feature extraction:
   - VI statistical features from preprocessed canopy regions.
   - DINOv3 ViT-S/16 deep embeddings.
6. Cross-field similarity and agronomic integration:
   - Controlled vs NoControl feature similarity.
   - Joint analysis with NDM and yield-related axes.
7. Multi-view 3D evaluation:
   - 24 view combinations (2 feature streams x 2 agronomic frameworks x 6 time points).
   - Z8 occupancy frequency used for ranking and candidate shortlist.

## Core Analytical Definitions

- Feature-space distance for genotype i at time t:
  d(i,t) = ||f_C(i,t) - f_N(i,t)||_2
- Similarity normalization:
  s(i,t) = 1 - (d(i,t)-d_min)/(d_max-d_min+eps)
- Time-aggregated similarity:
  s_bar(i) = mean_t s(i,t)
- Yield gain rate (stabilized):
  g(i) = (Y_N(i) - Y_C(i)) / (Y_C(i) + tau)

Where:
- f_C and f_N are Controlled and NoControl feature vectors.
- Y_C and Y_N are Controlled and NoControl yield values.

## Main Scripts and Roles

- experiments/image_and_label_preparation.py
  - Builds plot-level image dataset and metadata from externally prepared geospatial inputs.
- experiments/extract_vi_dinov3.py
  - Extracts VI and DINOv3 features.
- experiments/run_insect_resistance_analysis.py
  - Runs ranking and visualization workflows.

## Reproducibility Boundary

This repository does not execute WebODM reconstruction or interactive QGIS polygon editing.
Those steps are treated as upstream external preprocessing.

## Notes for Manuscript Alignment

- Keep manuscript terminology consistent with:
  - Controlled / NoControl
  - NDM, GY, Yield Gain Rate
  - Z1-Z8 partitioning and Z8 occupancy frequency
- If equations or thresholds evolve, update both manuscript and this document together.
