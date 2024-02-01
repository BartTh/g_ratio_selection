import glob
import os
import numpy as np
import nibabel as nib

from PIL import Image
from pathlib import Path

raw_data_dir = os.path.join(r'D:\Projects\published\g_ratio_selection\data\reformat\images')  # Input data directory

# Identity affine for Nifti image creation
affine = np.eye(4)
single_file_iteration = False

# Iterate over all jpg images in the raw data directory
for file in glob.glob(fr'{raw_data_dir}\*].jpg'):
    # Check if the file pertains to a mouse and the corresponding .nii.gz file does not exist
    if not Path.exists(Path(raw_data_dir).joinpath(f'{Path(file).name[:-4]}.nii.gz')):
        # Load the image file
        img = np.array(Image.open(file))[:4096, :4096]
        Image.fromarray(img).save(os.path.join(raw_data_dir, f'{Path(file).name[:-4]}-crop.jpg'))

        print(f'working on file {file}')
        # Convert the image to Nifti format
        nib_img = nib.Nifti1Image(img, affine=affine)
        # Save the Nifti image
        nib.save(nib_img, os.path.join(raw_data_dir, f'{Path(file).name[:-4]}-crop.nii.gz'))

        nib_lab = nib.Nifti1Image(img[:4096, :4096, 0] * 0, affine=affine)
        nib.save(nib_lab, os.path.join(Path(raw_data_dir).parent.joinpath('labels'),
                                       f'{Path(file).name[:-4]}-crop-label.nii.gz'))

        # If single file iteration flag is set, break the loop after first iteration
        if single_file_iteration:
            break
    else:
        print(f'already prepared .. {file}')
