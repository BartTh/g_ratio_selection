import glob
import json
import os
import shutil

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
from ..pipeline.morphometrics_utils import ConvertToMultiChanneld

experiment_name = "221219_2144_bz10_patch512_lr1e-03_e1600_chan16-qp4096-1"

root_dir = os.getcwd()
data_dir = os.path.join(root_dir, "Data")
proj_dir = os.path.join(root_dir, "experiments", experiment_name)

shutil.copy(os.path.join(root_dir, "eval.py"), proj_dir)

with open(os.path.join(proj_dir, "parameters.json")) as file:
    params = json.load(file)

files = {
    'test': {
        'image': sorted(glob.glob(os.path.join(data_dir, 'test', "images", "*.nii.gz"))),
        'label': sorted(glob.glob(os.path.join(data_dir, 'test', "labels", "*label.nii.gz")))
    }
}

set_files = {
    'test': [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(files['test']['image'], files['test']['label'])
    ]
}

test_transforms = Compose([
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
])

dice_metric = DiceMetric(include_background=True, reduction="mean")
post_trans = Compose([AsDiscrete(threshold=0.7)])

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

test_ds = CacheDataset(data=set_files['test'], transform=test_transforms, cache_rate=1.0, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

model.load_state_dict(
    torch.load(os.path.join(proj_dir, "best_during_training.pth"), map_location=device)
)
model.eval()

performance_store = {
    'Experiment': params['experiment_name'], 'Patient ID': [], 'Dice': [],
    'Accuracy': [], 'gt_vol': [], 'seg_vol': []
}

save_dir = os.path.join(root_dir, 'experiments', params['experiment_name'], 'predictions')
saver = SaveImage(output_dir=save_dir, output_ext=".png")

roi_size = (320, 320)
sw_batch_size = 4

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        pat_id = os.path.basename(test_data['image_meta_dict']['filename_or_obj'][0]).split('_')[0]
        print(f'Processing patient ID: {pat_id}')

        test_outputs = sliding_window_inference(
            inputs=test_data["image"].to(device),
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=0.5,
            mode='gaussian'
        )
        test_outputs = post_trans(test_outputs)

        for count, test_output in enumerate(test_outputs):
            filename = test_data["image_meta_dict"]['filename_or_obj'][count]
            name_dict = {'filename_or_obj': filename}
            saver(test_output, name_dict)

            f_name = name_dict['filename_or_obj'].split('\\')[-1][:-7]
            saved_file = os.path.join(save_dir, f_name, f"{f_name}_trans.png")

            im = np.swapaxes(np.array(Image.open(saved_file)), 0, 1)
            Image.fromarray((im * 255).astype("uint8")).convert('L').save(
                os.path.join(save_dir, f"{f_name}.png")
            )

        test_labels = test_data["label"].to(device)

        dice_value = np.round(dice_metric(y_pred=test_outputs, y=test_labels)[0].to('cpu'), 4)
        print(f"Dice in test set no {os.path.split(test_data['label_meta_dict']['filename_or_obj'][0])[1]}: "
              f"{dice_value}")
        performance_store['Dice'].append(dice_value)

        conf_matrix = get_confusion_matrix(y_pred=test_outputs, y=test_labels)
        accuracy = compute_confusion_matrix_metric('accuracy', conf_matrix).to('cpu')
        performance_store['Accuracy'].append(accuracy)

        performance_store['Patient ID'].append(pat_id)

mean_dice = str(np.mean([x.__array__() for x in performance_store['Dice']], axis=0))
mean_accuracy = str(np.mean([x.__array__() for x in performance_store['Accuracy']], axis=0))

print(f"Mean Dice: {mean_dice}")
print(f"Mean Accuracy: {mean_accuracy}")

performance_summary_path = os.path.join(os.path.dirname(save_dir), f"{mean_dice}.txt")
with open(performance_summary_path, 'w') as f:
    f.write(f"Mean Dice: {mean_dice}\n")
    f.write(f"Mean Accuracy: {mean_accuracy}")
