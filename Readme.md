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

```bash
python -m src.data.create_train_data
python -m src.training.train
python -m src.training.eval
python -m src.pipeline.morphometrics_pipeline
```

The pipeline stages in order:

1. **Data preparation** — `src/data/create_train_data.py`: converts raw JPG/PNG microscopy images and labels to NIfTI format (`.nii.gz`).

2. **Training** — `src/training/train.py`: trains a 2D UNet (MONAI) on the prepared dataset with Dice loss and TensorBoard logging.

3. **Inference and analysis** — `src/pipeline/morphometrics_pipeline.py`: runs sliding-window segmentation on new images and extracts per-fibre morphometrics and g-ratios. Use `morphometrics_pipeline_cluster.py` on HPC environments.

4. **Evaluation** — `src/training/eval.py`: computes Dice and accuracy metrics against ground truth labels.

5. **Group comparison** — `src/analysis/group_compare_service.py`: statistical comparison of g-ratio metrics across experimental groups with Tukey HSD correction.

**Note**: Dataset paths and model experiment names are configured directly within each script. Review and adjust these before running.

## **Citation**

If you use any part of the contents of this repository in your own work, please cite our paper:

Thomson, Bart R., et al. "Automated pipeline for nerve fiber selection and g-ratio calculation in optical microscopy: exploring staining protocol variations." Frontiers in Neuroanatomy 17 (2023): 1260186.

## **Contact**

If you have any questions or comments about this repository, please contact us at bart.thomson@uzh.ch.
