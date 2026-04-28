# Automated Pipeline for Nerve Fiber Selection and G-Ratio Calculation in Optical Microscopy: Exploring Staining Protocol Variations

## **Overview**

This GitHub repository contains the code and data used for our research paper entitled _Automated Pipeline for Nerve Fiber Selection and G-Ratio Calculation in Optical Microscopy: Exploring Staining Protocol Variations_. 

## Repository Structure

```
g_ratio_selection/
├── src/
│   ├── data/
│   │   └── create_train_data.py          # Convert raw images to NIfTI for training
│   ├── training/
│   │   ├── train.py                      # Train the UNet segmentation model
│   │   └── eval.py                       # Evaluate model performance on test set
│   ├── pipeline/
│   │   ├── morphometrics_pipeline.py     # Inference + g-ratio extraction (local)
│   │   ├── morphometrics_pipeline_cluster.py  # Inference + g-ratio extraction (HPC)
│   │   └── morphometrics_utils.py        # Segmentation transforms and morphometrics
│   └── analysis/
│       ├── group_compare_service.py      # Statistical group comparison and visualisation
│       └── count_nerves_in_gt.py         # Count nerve structures in ground truth images
└── tests/
    └── test_data.py                      # Validate NIfTI label shape and integrity
```

## Usage

Scripts use relative imports and must be run as modules from the repository root:

The pipeline stages in order:

1. **QuPath export** — run the QuPath script within QuPath to export tile images and labels from the whole-slide image. Then run `combine_tiles.py` to merge the exported tiles into the input format expected by the next step.

2. **Data preparation** — `src/data/create_train_data.py`: converts the combined JPG/PNG tile images and labels to NIfTI format (`.nii.gz`).

3. **Training** — `src/training/train.py`: trains a 2D UNet (MONAI) on the prepared dataset with Dice loss and TensorBoard logging.

4. **Inference and analysis** — `src/pipeline/morphometrics_pipeline.py`: runs sliding-window segmentation on new images and extracts per-fibre morphometrics and g-ratios. Use `morphometrics_pipeline_cluster.py` on HPC environments.

5. **Group comparison** — `src/analysis/group_compare.py`: statistical comparison of g-ratio metrics across experimental groups with Tukey HSD correction.

**Note**: Dataset paths and model experiment names are configured directly within each script. Review and adjust these before running.