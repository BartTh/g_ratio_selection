import copy
import glob
import shutil
import json
import torch
import cv2
import os
import numpy as np
import nibabel as nib

from PIL import Image
from pathlib import Path
from datetime import datetime
from morphometrics_utils import NerveMorphometrics, expand_myelin
from natsort import natsorted

from monai.data import DataLoader, CacheDataset
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsChannelFirstd,
    LoadImaged,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    ToTensord,
    Compose,
    AsDiscrete,
    SaveImage,
)

Image.MAX_IMAGE_PIXELS = None

# Flag to control whether the script should stop after processing a single file
single_file_iteration = False

# Dataset identifier
dataset = 'ppd_2000'

# Directory paths for raw data and other data related to the project
root_dir = os.getcwd()  # Current working directory
raw_data_dir = os.path.join(root_dir, 'data', 'input', f'{dataset}')  # Input data directory
data_dir = Path(os.path.join(root_dir, 'data', 'output', f'{dataset}'))  # Output data directory

inference_dir = Path(data_dir).joinpath('predictions')
root_dir = os.getcwd()

#methods paper model:
proj_dir = os.path.join(root_dir, 'experiments', '221219_2144_bz10_patch512_lr1e-03_e1600_chan16-qp4096-1')

# new_model
# proj_dir = os.path.join(root_dir, 'experiments', '231219_1346_bz10_patch512_lr1e-03_e1600_chan16-qp4096_0_05-1')

# Creating directories if they do not exist
Path.mkdir(inference_dir, exist_ok=True, parents=True)
Path.mkdir(Path(data_dir, 'images'), parents=True, exist_ok=True)

# Identity affine for Nifti image creation
affine = np.eye(4)

# Iterate over all jpg images in the raw data directory
for file in natsorted(glob.glob(fr'{raw_data_dir}\*\*.jpg')):
    # Check if the file pertains to a mouse and the corresponding .nii.gz file does not exist
    if not Path.exists(Path(data_dir).joinpath('images', f'{Path(file).name[:-4]}.nii.gz')):
        # Load the image file
        img = np.array(Image.open(file))
        # CHECK how to exclude the false segmentations
        # file_dir = r'D:\Projects\published\g_ratio_selection\data\output\20231213\images'
        # file1 = r'\AA1M3 P-3 Wdh [d=0.21717,x=34931,y=58219,w=1955,h=1954].jpg'
        # file2 = r'\AA1M3 P-3 Wdh [d=0.21717,x=36872,y=54338,w=1955,h=1954].jpg'
        # im1 = np.array(Image.open(file_dir + file1))
        # im2 = np.array(Image.open(file_dir + file2))
        # for i in range(0, 3):
        #     print(i)
        #     print(np.quantile(im1[:, :, i], 0.01))
        #     print(np.quantile(im2[:, :, i], 0.01))
        # print(np.quantile(im1[:, :], 0.01))
        # print(np.quantile(im2[:, :], 0.01))

        # Check if the image is not mostly 'empty'
        # by a rudimentary check on mean intensity (increase for more inclusion)
        print(img.shape, np.mean(img))
        if np.mean(img) < 250:
            print(np.quantile(img, 0.01))
            if np.quantile(img, 0.01) < 90:
                print(f'working on file {file}')
                # Convert the image to Nifti format
                nib_img = nib.Nifti2Image(img, affine=affine)
                # Save the Nifti image
                nib.save(nib_img, os.path.join(data_dir, 'images', f'{Path(file).name[:-4]}.nii.gz'))
                # Copy the original image to the new location
                shutil.copy2(file, os.path.join(data_dir, 'images'))
                # If single file iteration flag is set, break the loop after first iteration
                if single_file_iteration:
                    break
            else:
                print(f'filled with noise.... {file}')
        else:
            print(f'probably empty .. {file}')
        if single_file_iteration:
            break

    else:
        print(f'already prepared .. {file}')

print('Finished preparing raw files..')

# Load the parameters for model configuration from JSON file
params_file = proj_dir + r'\parameters.json'
params = json.load(open(params_file))

# Test transformations setup
test_transformations = [
    LoadImaged(keys=["image"]),
    AsChannelFirstd(keys=['image'], channel_dim=-1),
    Spacingd(keys=["image"], pixdim=(1, 1), mode='bilinear'),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"],
                         a_min=params['intensity_min'], a_max=params['intensity_max'],
                         b_min=0.0, b_max=1.0,
                         clip=True),
    ToTensord(keys=["image"]),
]

test_transforms = Compose(test_transformations)

# Define post-processing transformations for inference results
post_trans = Compose([AsDiscrete(threshold=0.7)])
post_label = Compose(AsDiscrete(to_onehot=params['n_classes']))

# Prepare the device, model, loss, and optimizer for PyTorch training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = UNet(
    dimensions=2,
    in_channels=3,
    out_channels=params['n_classes'],
    channels=params['num_channels'],
    strides=params['num_strides'],
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])

# Load the best model state from the previous training
model.load_state_dict(
    torch.load(os.path.join(proj_dir, "best_during_training.pth"),
               map_location=torch.device("cpu")))
model.eval()

# Define an object to save the inference results as images
saver = SaveImage(output_dir=inference_dir,
                  output_ext=".png")

# Get a sorted list of all Nifti image files to be used for inference
files = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))

# Iterate through the list of files which contain the images for inference.
for image_name in files:
    # Extract the file name without the path and extension.
    f_name = image_name.split('\\')[-1][:-7]

    # Skip processing if the transformed image already exists.
    if not Path(inference_dir.joinpath(f_name, f_name + '_trans.png')).exists():
        # Prepare the image data for the test dataset.
        set_files = [{"image": image_name}]

        # CacheDataset is used to load and transform the data once at the beginning to speed up processing.
        test_ds = CacheDataset(data=set_files, transform=test_transforms, cache_rate=1.0, num_workers=0)
        # DataLoader is used to load the dataset into batches, here batch_size is 1 as we process images one by one.
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

        # Disable gradient calculations as we are only performing inference, not training.
        with torch.no_grad():
            # Loop through the dataset and perform inference on each image using the model.
            for i, test_data in enumerate(test_loader):
                # Extract patient ID from the metadata of the image.
                pat_id = os.path.split(test_data['image_meta_dict']['filename_or_obj'][0])[1].split('_')[0]
                print(f'processing pat_id: {pat_id}')

                # Set the size of the region of interest and batch size for the sliding window inference.
                roi_size = (320, 320)
                sw_batch_size = 4

                # Perform inference with the sliding window technique.
                test_outputs = sliding_window_inference(
                    test_data["image"].to(device),
                    roi_size,
                    sw_batch_size,
                    model,
                    overlap=0.5,
                    mode='gaussian'
                )

                # Apply post-processing transformations to the output of the model.
                test_outputs = post_trans(test_outputs)

                # Save each output slice using the SaveImage utility.
                for count, test_output in enumerate(test_outputs):
                    name_dict = {'filename_or_obj': test_data["image_meta_dict"]['filename_or_obj'][count]}
                    saver(test_output, name_dict)
                    f_name = name_dict['filename_or_obj'].split('\\')[-1][:-7]
                    file = os.path.join(inference_dir, f_name, f_name + '_trans.png')

                    # Modify and save the image in the correct orientation and format.
                    im = np.swapaxes(np.array(Image.open(file)), 0, 1)
                    save_im = Image.fromarray((im * 255).astype("uint8")).convert('L')
                    save_im.save(os.path.join(inference_dir, f"{f_name}.png"))

        # If only a single file iteration is required, break the loop.
        if single_file_iteration:
            break
    else:
        # Inform the user that the file has already been processed.
        print(f"{inference_dir.joinpath(f_name, f_name + '_trans.png')} exists..")

# Create a directory to save the processed data files.
data_dir.joinpath('g_ratio_datafiles').mkdir(parents=True, exist_ok=True)

# Load all PNG files from the inference directory for processing.
files = [x for x in os.listdir(inference_dir) if x.endswith('.png')][:]
print(f'iterating over {len(files)} files..')

# Define parameters for morphometric analysis.
now = datetime.now().strftime("%Y%m%d")
axon_myelin_pixel_values = [127, 255, 127, 255]

# Iterate over each image for further processing and analysis.
for image_name in range(0, len(files)):
    # Construct the original image file path.
    im_name = data_dir.joinpath('images', files[image_name][:-4] + '.jpg')

    # Check if the segmented and selected image already exists to skip processing.
    if not data_dir.joinpath(
            f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_seg_selected.jpg').exists():
    # if not data_dir.joinpath('g_ratio_datafiles', f'Axon_seg-{files[image_name][:-4]}.csv').exists():
        print(f'Working on image {files[image_name][:-4]}')

        # Read the image file and convert it from BGR to RGB color space for processing
        im = cv2.cvtColor(cv2.imread(im_name.__str__()), cv2.COLOR_BGR2RGB)

        # Read the corresponding segmented image file in grayscale
        seg = inference_dir.joinpath(files[image_name])
        seg_im = cv2.imread(seg.__str__(), cv2.COLOR_BGR2GRAY)

        # Modify the segmentation image: absorb regions of uncertainty
        seg_im[seg_im == 76] = 0

        # Map specific pixel values to axon and myelin values for clarity in the segmentation
        seg_im[(seg_im == 150) | (seg_im == 179) | (seg_im == 226)] = axon_myelin_pixel_values[0]
        seg_im[(seg_im == 29) | (seg_im == 105)] = axon_myelin_pixel_values[1]

        # Expand myelin regions within the segmentation for accurate representation
        seg_im = expand_myelin(seg_im, axon_myelin_pixel_values)

        # Create an object for morphometric analysis of the segmented nerve image
        nerve_morphs = NerveMorphometrics(seg_im, axon_myelin_pixel_values)

        # Extract region properties from the segmentation
        filtered_props_df, raw_label_img = nerve_morphs.extract_region_props()

        try:
            # Find the center of mass (COM) for the nerves and select them based on a distance criterion
            final_df, seg_im_selected = nerve_morphs.find_com()
        except TypeError:
            # Handle the case where no nerves are found
            print('TypeError!')
            print(f'no nerves found in slide {files[image_name]}')
            continue
        except KeyError:
            # Handle the case where there is a KeyError in the processing
            print('KeyError')
            continue

        if not final_df:
            # If final_df is empty, no nerves have been found
            print('Finaldf Empty!')
            print(f'no nerves found in slide {files[image_name]}')
            continue

        # Calculate the g-ratio, which is an important metric for nerve fibers
        final_df = nerve_morphs.calculate_gratio()
        final_df['Myelin_seg'] = final_df['Myelin_seg'][final_df['Axon_seg'].Myelin_Thickness > 0]
        final_df['Axon_seg'] = final_df['Axon_seg'][final_df['Axon_seg'].Myelin_Thickness > 0]

        # Export the g-ratio data to CSV files
        for key in final_df:
            final_df[key].to_csv(data_dir.joinpath('g_ratio_datafiles', f'{key}-{files[image_name][:-4]}.csv'))

        if not final_df:
            # If final_df is still empty, no nerves have been found
            print(f'no nerves found in slide {files[image_name]}')
            continue

        # Create a directory for exporting images if it does not already exist
        data_dir.joinpath(f'exported_images/{files[image_name][:-4]}').mkdir(exist_ok=True, parents=True)

        # Convert the OpenCV image to a PIL Image and save it
        im = Image.fromarray(im)
        im.save(data_dir.joinpath(f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_im.jpg'))

        # Convert the segmentation image to a PIL Image and save it
        seg_im = Image.fromarray(seg_im)
        seg_im.save(data_dir.joinpath(f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_seg.jpg'))

        # Convert the selected segmentation image to a PIL Image and save it
        seg_im_selected = Image.fromarray(seg_im_selected)
        seg_im_selected.save(
            data_dir.joinpath(f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_seg_selected.jpg'))

        # If the script is set to only iterate over a single file, break the loop after processing
        if single_file_iteration:
            break
    else:
        # If the processed image already exists, print a message indicating so
        print(
            f"{data_dir.joinpath(f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_seg_selected.jpg')} "
            f"has been processed..")