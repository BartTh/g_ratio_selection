import glob
import os
import numpy as np
import nibabel as nib

from PIL import Image
from pathlib import Path

raw_data_dir = os.path.join(r'G:\My Drive\share\g-ratio\train\F43_0.05_4096')  # Input data directory

# Identity affine for Nifti image creation
affine = np.eye(4)
single_file_iteration = False

# Iterate over all jpg images in the raw data directory
for file in glob.glob(fr'{raw_data_dir}\*].jpg'):
    # Check if the file pertains to a mouse and the corresponding .nii.gz file does not exist
    if not Path.exists(Path(raw_data_dir).joinpath(f'{Path(file).name[:-4]}.nii.gz')):
        # Load the image file
        img = np.array(Image.open(file))
        lab = np.array(Image.open(file.split('.jpg')[0] + '.png'))

        print(f'working on file {file}')
        # Convert the image to Nifti format
        nib_img = nib.Nifti1Image(img, affine=affine)
        # Save the Nifti image
        nib.save(nib_img, os.path.join(str(Path(raw_data_dir).parent.joinpath('images')), f'{Path(file).name[:-4]}.nii.gz'))

        # Convert the label to Nifti format
        nib_lab = nib.Nifti1Image(lab, affine=affine)
        # Save the Nifti label
        nib.save(nib_lab, os.path.join(str(Path(raw_data_dir).parent.joinpath('labels')), f'{Path(file).name[:-4]}-label.nii.gz'))

        # If single file iteration flag is set, break the loop after first iteration
        if single_file_iteration:
            break
    else:
        print(f'already prepared .. {file}')