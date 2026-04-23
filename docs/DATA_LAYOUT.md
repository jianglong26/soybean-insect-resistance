# Data Layout (External Storage)

This repository does not include the full dataset because it is too large for GitHub.

External preprocessing is required before running this repository:

1. Build georeferenced orthomosaics from raw UAV RGB images (WebODM).
2. Co-register and delineate plot polygons in QGIS using RTK/GCP support.
3. Prepare phenotypic trait table (CSV/Excel) for label mapping.

Set SOY_DATA_ROOT to a folder that contains this structure:

AnhumasPiracicaba/
  dataset/
    annotations/
      dataset_metadata.json
    images/
      control/
      nocontrol/

Typical upstream assets stored outside Git may include:

AnhumasPiracicaba/
  orthomosaic/
    control/
    nocontrol/
    coordinates_*.csv
    *.qgz
  shapefiles/
    control/
    nocontrol/
  dataset/
    annotations/
      dataset_metadata.json
    images/
      control/
      nocontrol/

If SOY_DATA_ROOT is empty, default path is:

<repo_root>/AnhumasPiracicaba

Recommended: keep data and checkpoints on another disk, and only keep code in GitHub.
