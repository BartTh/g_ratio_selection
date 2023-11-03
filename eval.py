# Standard library imports
import glob
import json
import os
import shutil

# Third-party library imports
import numpy as np
import torch
from PIL import Image
from monai.data import DataLoader, CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_confusion_matrix_metric, get_confusion_matrix
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
    SaveImage
)
from morphometrics_utils import ConvertToMultiChanneld

# Experiment details
experiment_name = "221219_2144_bz10_patch512_lr1e-03_e1600_chan16-qp4096-1"

# Directory setup
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, "Data")
proj_dir = os.path.join(root_dir, "experiments", experiment_name)

# Copy evaluation script to the project directory
shutil.copy(os.path.join(root_dir, "eval.py"), proj_dir)

# Load parameters from the JSON configuration file
params_file = os.path.join(proj_dir, "parameters.json")
with open(params_file) as file:
    params = json.load(file)

# File paths setup
files = {
    'test': {
        'image': sorted(glob.glob(os.path.join(data_dir, 'test', "images",  "*.nii.gz"))),
        'label': sorted(glob.glob(os.path.join(data_dir, 'test', "labels", "*label.nii.gz")))
    }
}

# Combine test images and labels
set_files = {
    'test': [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(files['test']['image'], files['test']['label'])
    ]
}

# Test transformations setup
test_transformations = [
    LoadImaged(keys=["image", "label"]),
    AsChannelFirstd(keys=['image'], channel_dim=-1),
    ConvertToMultiChanneld(keys="label"),
    Spacingd(keys=["image", "label"], pixdim=(1, 1), mode=('bilinear', 'nearest')),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=params['intensity_min'], a_max=params['intensity_max'],
        b_min=0.0, b_max=1.0, clip=True
    ),
    ToTensord(keys=["image", "label"]),
]

# Compose transformations
test_transforms = Compose(test_transformations)

# Metrics and post-transforms setup
dice_metric = DiceMetric(include_background=True, reduction="mean")
post_trans = Compose([AsDiscrete(threshold=0.7)])
post_label = Compose([AsDiscrete(to_onehot=params['n_classes'])])

# Setup UNet and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet(
    dimensions=2,
    in_channels=3,
    out_channels=params['n_classes'],
    channels=params['num_channels'],
    strides=params['num_strides'],
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

# Prepare test dataset and data loader
test_ds = CacheDataset(data=set_files['test'], transform=test_transforms, cache_rate=1.0, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

# Metrics initialization
metric_count = 0
metric_sum = 0

# Load saved model
model.load_state_dict(
    torch.load(os.path.join(proj_dir, "best_during_training.pth"), map_location=device)
)
model.eval()

# Initialize performance store
performance_store = {
    'Experiment': params['experiment_name'], 'Patient ID': [], 'Dice': [],
    'Accuracy': [], 'gt_vol': [], 'seg_vol': []
}

# Prepare directory for saving predictions
save_dir = os.path.join(root_dir, 'experiments', params['experiment_name'], 'predictions')

# Initialize the image saver
saver = SaveImage(output_dir=save_dir, output_ext=".png")

# Inference
# Define ROI size and the batch size for sliding window
roi_size = (320, 320)
sw_batch_size = 4

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        # Extract patient ID from the file path
        pat_id = os.path.basename(test_data['image_meta_dict']['filename_or_obj'][0]).split('_')[0]
        print(f'Processing patient ID: {pat_id}')

        # Perform inference using the sliding window method
        test_outputs = sliding_window_inference(
            inputs=test_data["image"].to(device),
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=0.5,
            mode='gaussian'
        )
        # Apply post-processing transformations
        test_outputs = post_trans(test_outputs)

        # Save output images
        for count, test_output in enumerate(test_outputs):
            # Prepare filename dictionary for saver
            filename = test_data["image_meta_dict"]['filename_or_obj'][count]
            name_dict = {'filename_or_obj': filename}
            saver(test_output, name_dict)

            # Construct the file path for saved images
            f_name = name_dict['filename_or_obj'].split('\\')[-1][:-7]
            saved_file = os.path.join(save_dir, f_name, f"{f_name}_trans.png")

            # Open the saved image, process it, and save the final image
            im = np.swapaxes(np.array(Image.open(saved_file)), 0, 1)
            save_im = Image.fromarray((im * 255).astype("uint8")).convert('L')
            final_path = os.path.join(save_dir, f"{f_name}.png")
            save_im.save(final_path)

        # Move the label to the correct device
        test_labels = test_data["label"].to(device)

        # Compute and store metrics for the current batch
        dice_value = np.round(dice_metric(y_pred=test_outputs, y=test_labels)[0].to('cpu'), 4)

        print(f"Dice in test set no { os.path.split(test_data['label_meta_dict']['filename_or_obj'][0])[1]}: "
              f"{dice_value}")

        performance_store['Dice'].append(dice_value)

        # Calculate and store accuracy
        conf_matrix = get_confusion_matrix(y_pred=test_outputs, y=test_labels)
        accuracy = compute_confusion_matrix_metric('accuracy', conf_matrix).to('cpu')

        performance_store['Accuracy'].append(accuracy)

        # Store the patient ID
        performance_store['Patient ID'].append(pat_id)

# Calculate and report mean dice and accuracy
mean_dice = str(np.mean([x.__array__() for x in performance_store['Dice']], axis=0))
mean_accuracy = str(np.mean([x.__array__() for x in performance_store['Accuracy']], axis=0))

print(f"Mean Dice: {mean_dice}")
print(f"Mean Accuracy: {mean_accuracy}")

# Save mean dice and accuracy to text
performance_summary_path = os.path.join(os.path.dirname(save_dir), f"{mean_dice}.txt")
with open(performance_summary_path, 'w') as f:
    f.write(f"Mean Dice: {mean_dice}\n")
    f.write(f"Mean Accuracy: {mean_accuracy}")
