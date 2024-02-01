import nibabel as nib
import glob

directory = r'D:\Projects\published\g_ratio_selection\data\validation\labels'


for file in glob.glob(r'D:\Projects\published\g_ratio_selection\data\*\labels\*-crop-label.nii.gz'):
    img = nib.load(file)
    print(file, '\n', img.shape)
    data = img.get_fdata()

    # Check if the data has the shape (4096, 4096, 3)
    if data.shape == (4096, 4096, 3):
        # Remove the last channel
        modified_data = data[:, :, 0]

        # Create a new NIfTI image
        new_img = nib.Nifti1Image(modified_data, img.affine)
        nib.save(new_img, file)
        print(new_img.shape)
