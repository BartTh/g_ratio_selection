import glob
import os
import shutil
import json

import numpy as np
import torch
from PIL import Image
from morphometrics_utils import ConvertToMultiChanneld

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
    SaveImage,
)


experiment_name = '221219_2144_bz10_patch512_lr1e-03_e1600_chan16-qp4096-1'

root_dir = os.getcwd()
data_dir = r'.\Data'
proj_dir = os.path.join(root_dir, 'experiments', experiment_name)

shutil.copy('eval.py', proj_dir)

params_file = proj_dir + r'\parameters.json'
params = json.load(open(params_file))

files = {'test': {}}
set_files = {}


files['test']['image'] = sorted(glob.glob(os.path.join(data_dir, 'test', "images", "*.nii.gz")))[:]
files['test']['label'] = sorted(glob.glob(os.path.join(data_dir, 'test', "labels", "*label.nii.gz")))[:]
test_transformations = [
    LoadImaged(keys=["image", "label"]),
    AsChannelFirstd(keys=['image'], channel_dim=-1),
    ConvertToMultiChanneld(keys="label"),
    Spacingd(keys=["image", "label"], pixdim=(1, 1), mode=('bilinear', 'nearest')),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=params['intensity_min'], a_max=params['intensity_max'], b_min=0.0,
                         b_max=1.0, clip=True),
    ToTensord(keys=["image", "label"]),
]

set_files['test'] = [{"image": image_name, "label": label_name} for image_name, label_name in
                     zip(files['test']['image'], files['test']['label'])]

test_transforms = Compose(test_transformations)

dice_metric = DiceMetric(include_background=True, reduction="mean")

post_trans = Compose([AsDiscrete(threshold=0.7)])
post_label = Compose(AsDiscrete(to_onehot=params['n_classes']))

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
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

test_ds = CacheDataset(data=set_files['test'], transform=test_transforms, cache_rate=1.0, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

metric_count = 0
metric_sum = 0
model.load_state_dict(
    torch.load(os.path.join(proj_dir, "best_during_training.pth"),
               map_location=torch.device("cpu")))
model.eval()
show_slice = 40

performance_store = {'Experiment': params['experiment_name'], 'Patient ID': [], 'Dice': [], 'Accuracy': [],
                     'gt_vol': [], 'seg_vol': []}

save_dir = os.path.join(root_dir, 'experiments', '{}'.format(params['experiment_name']), 'predictions')
saver = SaveImage(output_dir=save_dir,
                  output_ext=".png")

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        pat_id = os.path.split(test_data['image_meta_dict']['filename_or_obj'][0])[1].split('_')[0]
        print(f'processing pat_id: {pat_id}')
        roi_size = (320, 320)
        sw_batch_size = 4
        test_outputs = sliding_window_inference(
            test_data["image"].to(device),
            roi_size,
            sw_batch_size,
            model,
            overlap=0.5,
            mode='gaussian'
        )
        test_outputs = post_trans(test_outputs)

        for count, test_output in enumerate(test_outputs):
            name_dict = {'filename_or_obj': test_data["image_meta_dict"]['filename_or_obj'][count]}
            saver(test_output, name_dict)
            f_name = name_dict['filename_or_obj'].split('\\')[-1][:-7]
            file = os.path.join(save_dir, f_name, f_name + '_trans.png')
            im = np.swapaxes(np.array(Image.open(file)), 0, 1)
            save_im = Image.fromarray((im * 255).astype("uint8")).convert('L')
            save_im.save(os.path.join(save_dir, f"{f_name}.png"))

        test_labels = test_data["label"].to(device)

        # compute metric for current iteration
        dice_metric(y_pred=test_outputs, y=test_labels)
        conf_matrix = get_confusion_matrix(y_pred=test_outputs, y=test_labels)

        dice_value = np.round(dice_metric(y_pred=test_outputs, y=test_labels)[0].to('cpu'), 4)

        print("Dice in test set no {}: {}".format(
            os.path.split(test_data['label_meta_dict']['filename_or_obj'][0])[1], dice_value))

        performance_store['Dice'].append(dice_value)
        performance_store['Accuracy'].append(compute_confusion_matrix_metric('accuracy', conf_matrix).to('cpu'))
        performance_store['Patient ID'].append(pat_id)

mean_dices = str(np.mean([x.__array__() for x in performance_store['Dice']], axis=0))
mean_acc = str(np.mean([x.__array__() for x in performance_store['Accuracy']], axis=0))

with open(fr'{os.path.split(save_dir)[0]}\{mean_dices}.txt', 'w') as f:
    f.write(f'{mean_dices}')
