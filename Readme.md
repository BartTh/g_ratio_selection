# Automated Pipeline for Nerve Fiber Selection and G-Ratio Calculation in Optical Microscopy: Exploring Staining Protocol Variations

## **Overview**

This GitHub repository contains the code and data used for our research paper entitled _Automated Pipeline for Nerve Fiber Selection and G-Ratio Calculation in Optical Microscopy: Exploring Staining Protocol Variations_. 

## **Content**

..

## Usage

The pipeline consists of several scripts designed for different stages of the analysis:

1. **Data Preparation**:

   - Use `create_train_data.py` to preprocess and format raw microscopy images for analysis.

2. **Training**:

   - Execute `train.py` to train the segmentation model on prepared datasets.

3. **Inference and Analysis**:

   - Run `morphometrics_pipeline.py` to perform segmentation on new images and extract morphometric data.

4. **Evaluation**:

   - Utilize `eval.py` to assess the performance of the segmentation model against ground truth data.
   - 
5. **Utility Scripts**:

   - `count_nerves_in_gt.py`: Counts the number of nerve structures in ground truth images.
   - `test.py`: Contains test functions for validating data integrity.

**Note**: Detailed instructions and parameter configurations for each script are provided within the scripts themselves. It's recommended to review and adjust these parameters based on your specific dataset and research requirements.

## **Citation**

If you use any part of the contents of this repository in your own work, please cite our paper:

Thomson, Bart R., et al. "Automated pipeline for nerve fiber selection and g-ratio calculation in optical microscopy: exploring staining protocol variations." Frontiers in Neuroanatomy 17 (2023): 1260186.

## **Contact**

If you have any questions or comments about this repository, please contact us at bart.thomson@uzh.ch.
