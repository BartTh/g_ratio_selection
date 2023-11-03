import cv2
import glob
from skimage.measure import label
import numpy as np

labs = []
for file in glob.glob(r'D:\Data\Gratio\labeled_data_0.05\*\labels\*-label*.png'):
    if not 'incomplete' in file:
        gt_im = cv2.imread(file.__str__(), cv2.COLOR_BGR2GRAY) == 127
        unique_labels = len(np.unique(label(gt_im)))
        labs.append(unique_labels)
        print(file, unique_labels, np.sum(labs))
print(np.min(labs), np.max(labs))
