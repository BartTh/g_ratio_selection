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


single_file_iteration = False
dataset = 'control_ppd'
raw_data_dir = fr'D:\Data\Gratio\control_paper\{dataset}'
data_dir = Path(fr'D:\Data\Gratio\vanilla\{dataset}')
inference_dir = Path(data_dir).joinpath('predictions')
root_dir = os.getcwd()
proj_dir = os.path.join(root_dir, 'experiments', '221219_2144_bz10_patch512_lr1e-03_e1600_chan16-qp4096-1')

Path.mkdir(inference_dir, exist_ok=True, parents=True)

Path.mkdir(Path(data_dir, 'images'), parents=True, exist_ok=True)
affine = np.eye(4)

for file in glob.glob(fr'{raw_data_dir}\*\*.jpg'):
    mouse_no = Path(file).parent.name[:3]
    if 'M' in mouse_no and not Path.exists(Path(data_dir).joinpath('images', f'{Path(file).name[:-4]}.nii.gz')):
        img = np.array(Image.open(file))

        if np.mean(img) < 200:
            print(f'working on file {file}')
            nib_img = nib.Nifti1Image(img, affine=affine)
            nib.save(nib_img, os.path.join(data_dir, 'images', f'{Path(file).name[:-4]}.nii.gz'))
            shutil.copy2(file, os.path.join(data_dir, 'images'))
        else:
            print(f'probably empty .. {file}')
        if single_file_iteration:
            break
    else:
        print(f'already prepared .. {file}')

print('Finished preparing raw files..')

params_file = proj_dir + r'\parameters.json'
params = json.load(open(params_file))

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

model.load_state_dict(
    torch.load(os.path.join(proj_dir, "best_during_training.pth"),
               map_location=torch.device("cpu")))
model.eval()

saver = SaveImage(output_dir=inference_dir,
                  output_ext=".png")

files = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))

for image_name in files:
    f_name = image_name.split('\\')[-1][:-7]
    if not Path(inference_dir.joinpath(f_name, f_name + '_trans.png')).exists():
        set_files = [{"image": image_name}]

        test_ds = CacheDataset(data=set_files, transform=test_transforms, cache_rate=1.0, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

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
                    file = os.path.join(inference_dir, f_name, f_name + '_trans.png')
                    im = np.swapaxes(np.array(Image.open(file)), 0, 1)
                    save_im = Image.fromarray((im * 255).astype("uint8")).convert('L')
                    save_im.save(os.path.join(inference_dir, f"{f_name}.png"))

        if single_file_iteration:
            break
    else:
        print(f"{inference_dir.joinpath(f_name, f_name + '_trans.png')} exists..")

data_dir.joinpath('g_ratio_datafiles').mkdir(parents=True, exist_ok=True)

pixel_size = 0.05

files = [x for x in os.listdir(inference_dir) if x.endswith('.png')][:]
print(f'iterating over {len(files)} files..')
max_com_distance = 20
now = datetime.now().strftime("%Y%m%d")
axon_myelin_pixel_values = [127, 255, 127, 255]
downsample = False

for image_name in range(0, len(files)):
    im_name = data_dir.joinpath('images', files[image_name][:-4] + '.jpg')
    try:
        mouse = im_name.name[:3]
        mouse_no = int(mouse[1:])
    except ValueError:
        mouse = im_name.name[:4]
        mouse_no = int(mouse[2:])

    if not data_dir.joinpath(f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_seg_selected.jpg').exists():
        print(f'Working on image {files[image_name][:-4]}')
        im = cv2.cvtColor(cv2.imread(im_name.__str__()), cv2.COLOR_BGR2RGB)

        seg = inference_dir.joinpath(files[image_name])
        seg_im = cv2.imread(seg.__str__(), cv2.COLOR_BGR2GRAY)

        if downsample:
            im = cv2.resize(im, dsize=(9000//downsample, 9000//downsample), interpolation=cv2.INTER_NEAREST)
            seg_im = cv2.resize(seg_im, dsize=(9000//downsample, 9000//downsample), interpolation=cv2.INTER_NEAREST)

        # absorb the regions of uncertainty
        seg_im[seg_im == 76] = 0
        seg_im[(seg_im == 150) | (seg_im == 179) | (seg_im == 226)] = axon_myelin_pixel_values[0]
        seg_im[(seg_im == 29) | (seg_im == 105)] = axon_myelin_pixel_values[1]

        seg_im = expand_myelin(seg_im, axon_myelin_pixel_values)

        nerve_morphs = NerveMorphometrics(seg_im, axon_myelin_pixel_values, pixel_size)

        filtered_props_df, raw_label_img = nerve_morphs.extract_region_props()
        try:
            final_df, seg_im_selected = nerve_morphs.find_com(max_com_distance=max_com_distance)
        except TypeError:
            print('TypeError!')
            print(f'no nerves found in slide {files[image_name]}')
            continue
        except KeyError:
            print('KeyError')
            continue

        if not final_df:
            print('Finaldf Empty!')
            print(f'no nerves found in slide {files[image_name]}')
            continue

        final_df = nerve_morphs.calculate_gratio()
        for key in final_df:
            final_df[key].to_csv(data_dir.joinpath('g_ratio_datafiles', f'{key}-{files[image_name][:-4]}.csv'))

        if not final_df:
            print(f'no nerves found in slide {files[image_name]}')
            continue

        data_dir.joinpath(f'exported_images/{files[image_name][:-4]}').mkdir(exist_ok=True, parents=True)

        im = Image.fromarray(im)
        im.save(data_dir.joinpath(f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_im.jpg'))
        seg_im = Image.fromarray(seg_im)
        seg_im.save(data_dir.joinpath(f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_seg.jpg'))
        seg_im_selected = Image.fromarray(seg_im_selected)
        seg_im_selected.save(data_dir.joinpath(f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_seg_selected.jpg'))

        if single_file_iteration:
            break
    else:
        print(f"{data_dir.joinpath(f'exported_images/{files[image_name][:-4]}/{files[image_name][:-4]}_seg_selected.jpg')} "
              f"has been processed..")

    _mouse_no = copy.deepcopy(mouse_no)
