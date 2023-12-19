import glob
import os
import shutil
import re
import datetime
import json
import itertools
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from morphometrics_utils import ConvertToMultiChanneld

from monai.data import DataLoader, CacheDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ToTensord,
    RandCropByLabelClassesd,
    RandScaleIntensityd,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandZoomd,
    RandFlipd,
    Spacingd,
    Activations,
    AsChannelFirstd,
    Compose,
    AsDiscrete,
)
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image

# Define parameters for the experiment
params = {
    'experiment_name': {},
    'qupath_export_and_scaling': "4096_9000_0_05",
    'patch_size': (512, 512),
    'bz_size': 10,
    'n_classes': 3,
    'class_ratios': [1, 10, 10],
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epoch_num': 2,
    'val_interval': 2,
    'num_channels': (16, 32, 64, 128, 256, 512),
    'num_strides': (2, 2, 2, 2, 2),
    'intensity_min': 0,
    'intensity_max': 255,
    'seed': 0,
    'train_set': {},
    'val_set': {},
    'test_set': {},
    'train_transforms': {},
    'val_transforms': {},
    'loss_function': DiceLoss(squared_pred=True, to_onehot_y=False, softmax=True)
}

# Generate a timestamp for the experiment name
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")[2:]

# Update the experiment name with formatted parameters
params['experiment_name'] = "{}_bz{}_patch{}_lr{:.0e}_e{}_chan{}-qp{}".format(
    now,
    params['bz_size'],
    params["patch_size"][0],
    params["learning_rate"],
    params["epoch_num"],
    params["num_channels"][0],
    params["qupath_export_and_scaling"]
)

# Set the seed for reproducibility
set_determinism(seed=params['seed'])

# Define the root directory for the data and the subdirectories
root_dir = os.getcwd()  # Current working directory
data_dir = os.path.join(root_dir, 'Data')  # Data directory

# Dictionary to hold file paths for each set
files = {}
image_sets = ['train', 'validation', 'test']

# Initialize the files dictionary for each image set
for set_name in image_sets:
    files[set_name] = {}

# Initialize a dictionary to store file paths for images and labels
set_files = {}

# File paths setup
for nn_set in image_sets:
    # Collect all .nii.gz image file paths within each image set
    files[nn_set]['image'] = sorted(glob.glob(os.path.join(data_dir, nn_set, "images", "*.nii.gz")))
    # Collect all corresponding label file paths
    files[nn_set]['label'] = sorted(glob.glob(os.path.join(data_dir, nn_set, "labels", "*label.nii.gz")))

    # Create a list of dictionaries with corresponding image and label paths
    set_files[nn_set] = [{"image": image_name, "label": label_name}
                         for image_name, label_name in zip(files[nn_set]['image'], files[nn_set]['label'])]

# Variable to append to the experiment name to avoid overwriting
index = '1'
# Loop until a unique directory is created
while True:
    try:
        # Attempt to create a directory for the current experiment
        os.makedirs(os.path.join(root_dir, 'experiments', '{}-{}'.format(params['experiment_name'], index)))
        # Update experiment name with the index if directory creation is successful
        params['experiment_name'] = '{}-{}'.format(params['experiment_name'], index)
        break
    except FileExistsError:
        # If directory already exists, increment the index and try again
        index = str(int(index) + 1)

# Define the project directory path
proj_dir = os.path.join(root_dir, 'experiments', params['experiment_name'])
print(proj_dir)

# Copy the training script to the project directory
shutil.copy('train.py', proj_dir)

# Create a 'predictions' directory within the experiment directory if it doesn't exist
predictions_dir = os.path.join(proj_dir, 'predictions')
if not os.path.exists(predictions_dir):
    os.mkdir(predictions_dir)

# Extract image names for each set using regular expressions
params['train_set'] = [re.search(r'images\\(.+?).nii.gz', x)[1] for x in files['train']['image']]
params['val_set'] = [re.search(r'images\\(.+?).nii.gz', x)[1] for x in files['validation']['image']]
params['test_set'] = [re.search(r'images\\(.+?).nii.gz', x)[1] for x in files['test']['image']]

# Train transformations setup
train_transformations = [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys=['image'], channel_dim=-1),
        ConvertToMultiChanneld(keys="label"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1), mode=('bilinear', 'nearest')),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=params['intensity_min'], a_max=params['intensity_max'],
                             b_min=0.0, b_max=1.0, clip=True),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=params['patch_size'],
            ratios=params['class_ratios'],
            num_classes=params['n_classes'],
            num_samples=4,
            image_threshold=0,
        ),
        RandGaussianSmoothd(keys=["image"], sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), approx='erf', prob=0.1),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
        RandZoomd(keys=["image"], prob=0.1, min_zoom=0.9, max_zoom=1.1),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
        RandFlipd(["image", "label"], spatial_axis=0, prob=0.5),
        ToTensord(keys=["image", "label"]),
    ]

# Validation transformations setup
val_transformations = [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys=['image'], channel_dim=-1),
        ConvertToMultiChanneld(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=params['intensity_min'], a_max=params['intensity_max'], b_min=0.0, b_max=1.0, clip=True),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=params['patch_size'],
            ratios=params['class_ratios'],
            num_classes=params['n_classes'],
            num_samples=4,
            image_threshold=0,
        ),
        ToTensord(keys=["image", "label"]),
    ]

# Compose the lists of transformations into a single transform
train_transforms = Compose(train_transformations)
val_transforms = Compose(val_transformations)

# Save the transformation parameters into the params dictionary
params['train_transforms'] = [vars(x) for x in train_transformations]
params['val_transforms'] = [vars(x) for x in val_transformations]

# Save parameters to JSON for eval
json_params = dict(itertools.islice(params.items(), 15))
with open(os.path.join(root_dir, 'experiments', f"{params['experiment_name']}", 'parameters.json'), 'w') as json_file:
    json.dump(json_params, json_file)

# Save all parameters to a text file for reference
with open(os.path.join(root_dir, 'experiments', f"{params['experiment_name']}", 'parameters.txt'), 'w') as txt_file:
    print(params, file=txt_file)

# Prepare datasets and data loaders for training and validation
train_ds = CacheDataset(data=set_files['train'], transform=train_transforms, cache_rate=1.0, num_workers=0)
train_loader = DataLoader(train_ds, batch_size=params['bz_size'], shuffle=True, num_workers=0)

val_ds = CacheDataset(data=set_files['validation'], transform=val_transforms, cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

# Example data visualizations to ensure that the images and labels have the correct shapes
train_data_example = train_ds[1][0]
val_data_example = val_ds[0][0]

# Visualize training and validation data
plt.figure("image", (18, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(val_data_example["image"][i, :, :], cmap="gray")
plt.show()

plt.figure("label", (18, 6))
for i in range(params['n_classes']):
    plt.subplot(1, 3, i + 1)
    plt.title(f"label channel {i}")
    plt.imshow(val_data_example["label"][i, :, :] == 1)
plt.show()

# Initialize metrics and post-processing transforms for model evaluation
dice_metric = DiceMetric(include_background=True, reduction="mean")
post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=params['n_classes'])])

# Set up model, loss, and optimizer
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

optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

# Training loop initialization
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter(log_dir=os.path.join(root_dir, 'experiments', f"{params['experiment_name']}", 'train'))

# Training and validation loop
for epoch in range(params['epoch_num']):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{params['epoch_num']}")
    model.train()
    epoch_loss = 0
    step = 0

    # Training step
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = params['loss_function'](outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # Validation step
    if (epoch + 1) % params['val_interval'] == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = params['n_classes'] * [0.0]
            metric_count = params['n_classes'] * [0]
            val_inputs = None
            val_labels = None
            val_outputs = None
            print('torch.no_grad')

            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                roi_size = (320, 320)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs,
                                                       roi_size,
                                                       sw_batch_size,
                                                       model,
                                                       mode='gaussian')
                val_outputs = post_pred(val_outputs)

                value = dice_metric(y_pred=val_outputs, y=val_labels)
                class_dice = [0 if x != x else x for x in value.to('cpu')[0].tolist()]
                metric_sum = np.sum([metric_sum, class_dice], axis=0)
                metric_count = np.sum([[0 if x != x else 1 for x in class_dice], metric_count], axis=0)
                break

            class_metric = metric_sum / metric_count
            mean_metric = np.mean(class_metric)
            metric_values.append(mean_metric)

            # Save model if performance improved
            if mean_metric > best_metric:
                best_metric = mean_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(),
                           os.path.join(root_dir, 'experiments', '{}'.format(params['experiment_name']),
                                        "best_during_training.pth"))
                print("saved new best metric model")
            print(f"current epoch: {epoch + 1} current mean dice: {mean_metric:.4f}, class dice: {class_metric}"
                  f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

            # Add validation metrics to TensorBoard
            writer.add_scalar("val_mean_dice", mean_metric, epoch + 1)
            for _class in range(len(class_dice)):
                writer.add_scalar(f"dice class {_class}", class_metric[_class].item(), epoch + 1)

            # Visualization of model predictions in TensorBoard
            plot_2d_or_3d_image(val_inputs, epoch + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

torch.save(model.state_dict(), os.path.join(root_dir, 'experiments', '{}'.format(params['experiment_name']),
                                            "dice_{}_at_{}.pth".format(np.round(best_metric, 4), best_metric_epoch)))
