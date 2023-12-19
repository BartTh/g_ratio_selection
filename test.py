import nibabel as nib

im = nib.load(r'D:\Projects\published\g_ratio_selection\data\test\labels\M01-Series5 [d=0.2,x=5645,y=8870,w=819,h=820]-label.nii.gz')
img = im.get_fdata()

im = nib.load(r'D:\Switchdrive\gratio\train\labels\M01-Series5 [d=0.2,x=5645,y=8870,w=819,h=820]-label.nii.gz')
img = im.get_fdata()
