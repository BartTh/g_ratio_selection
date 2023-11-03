import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from skimage import util
from skimage.morphology import disk
import copy
from skimage.measure import label, regionprops_table
from monai.transforms import MapTransform


class ConvertToMultiChanneld(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 0 is the background
    label 1 is the axon
    label 2 is the myelin
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 0)
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


def expand_myelin(seg_im, axon_myelin_pixel_values):
    """
    Expand myelin regions in the segmented image
    """

    myel_im = copy.deepcopy(seg_im)
    seg_im[seg_im == axon_myelin_pixel_values[1]] = 0
    myel_im[myel_im != axon_myelin_pixel_values[1]] = 0

    kernel = disk(2)
    myel_im = cv2.dilate(myel_im, kernel)
    seg_im[myel_im != 0] = axon_myelin_pixel_values[1]

    return seg_im


def is_outlier(points, thresh=3.5):
    """
    Returns a list of points that are considered outliers.

    Parameters:
        points: An numobservations by numdimensions array of observations
        thresh: The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
        list: List of integers that represent outliers in the points input parameter

    References:
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        Kington, J. D., and H. J. Tobin. "Balanced cross sections,
        shortening estimates, and the magnitude of out-of-sequence thrusting in
        the Nankai Trough accretionary prism, Japan." AGU Fall Meeting Abstracts. Vol. 2011. 2011.
    """
    points = np.array(points, dtype=int)
    actual_points = len(points) // 2

    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    isoutlier = modified_z_score > thresh
    outlier_idx_vert = [i for i, x in enumerate(isoutlier) if x and i < actual_points]
    outlier_idx_horz = [i - actual_points for i, x in enumerate(isoutlier) if x and i > actual_points]

    return np.unique(np.sort(np.concatenate([outlier_idx_vert, outlier_idx_horz]).astype('int')))


class NerveMorphometrics:
    def __init__(self, seg_im, axon_myelin_pixel_values, pixel_size):
        """
        Initialize NerveMorphometrics class

        Parameters:
            seg_im (numpy array): The segmented image of the axons and myelin.
            axon_myelin_pixel_values (list): The pixel values that correspond to axons and myelin in the
            segmented and ground truth images.
        """
        self.props_df = {'Axon_seg': pd.DataFrame(), 'Myelin_seg': pd.DataFrame()}
        self.seg_im = seg_im
        self.seg_im_selected = copy.deepcopy(seg_im)
        self.raw_label_img = {}
        self.filtered_props_df = {}
        self.final_df = {'Axon_seg': pd.DataFrame(), 'Myelin_seg': pd.DataFrame()}
        self.axon_myelin_pixel_values = axon_myelin_pixel_values
        self.pixel_size = pixel_size

    def extract_region_props(self):
        """
        Identify morphometrics of the axons and myelin in the segmented and ground truth images.

        Returns:
            final_df (dict): containing the filtered properties dataframe of the axons and myelin in the segmented image
            gt_im_selected: the labeled image of the axons and myelin in the ground truth image
            seg_im_selected: the labeled image of the axons and myelin in the segmented image
        """
        for idx, label_id in enumerate(['Axon_seg', 'Myelin_seg']):
            if 'seg' in label_id:
                binary_im = np.zeros(self.seg_im.shape)
                binary_im[self.seg_im == self.axon_myelin_pixel_values[idx]] = 1

            self.raw_label_img[label_id] = label(binary_im, connectivity=1)

            props = regionprops_table(self.raw_label_img[label_id],
                                      properties=('label', 'centroid', 'eccentricity', 'bbox',
                                                  'solidity', 'area_filled', 'equivalent_diameter_area'))

            self.props_df[label_id] = pd.DataFrame(props)

        self.filtered_props_df = {'Axon_filt': self.props_df['Axon_seg'][
            (self.props_df['Axon_seg']['eccentricity'] < 0.95) &
            (self.props_df['Axon_seg']['solidity'] > 0.9) &
            (self.props_df['Axon_seg']['area_filled'] > 50) &
            (self.props_df['Axon_seg']['bbox-0'] != 0) &
            (self.props_df['Axon_seg']['bbox-1'] != 0) &
            (self.props_df['Axon_seg']['bbox-2'] != self.seg_im.shape[0]) &
            (self.props_df['Axon_seg']['bbox-3'] != self.seg_im.shape[1])
        ],
                                  'Axon_seg': [],
                                  'Myelin_seg': []}
        return self.filtered_props_df, self.raw_label_img

    def find_com(self, max_com_distance=20):
        """
        Find the center of mass (CoM) of the myelin segments within the bounding box of the filtered axon segments.

        Parameters:
        max_com_distance (int): The maximum distance between the CoM of the myelin segments and the bounding box
        of the filtered axon segments.

        Returns:
        dict: A dictionary containing the filtered properties dataframe of the axon and myelin segments,
        with the myelin segments' CoM within the bounding box of the axon segments.
        """
        # Find myelin CoM within axon seg bbox
        for mye_index, mye_row in self.props_df['Myelin_seg'].iterrows():
            for axon_idx, axon_row in self.filtered_props_df['Axon_filt'].iterrows():
                if axon_row['bbox-0'] < mye_row['centroid-0'] < axon_row['bbox-2'] and \
                        axon_row['bbox-1'] < mye_row['centroid-1'] < axon_row['bbox-3']:
                    self.filtered_props_df['Myelin_seg'].append(mye_row)
                    self.filtered_props_df['Axon_seg'].append(axon_row)

        self.filtered_props_df['Myelin_seg'] = pd.DataFrame(self.filtered_props_df['Myelin_seg']).reset_index(drop=True)  # .drop_duplicates()
        self.filtered_props_df['Axon_seg'] = pd.DataFrame(self.filtered_props_df['Axon_seg']).reset_index(drop=True)  # .drop_duplicates()

        drop_non_closed_myelin = []
        # check for myelin endpoints and remove based on number of endpoints
        for mye_index, myelin_row in self.filtered_props_df['Myelin_seg'].iterrows():
            myelin_of_interest = self.seg_im_selected[int(myelin_row['bbox-0']): int(myelin_row['bbox-2']),
                                 int(myelin_row['bbox-1']): int(myelin_row['bbox-3'])]
            label_mye_interest = label(myelin_of_interest == self.axon_myelin_pixel_values[1], connectivity=1)
            mye_props = regionprops_table(label_mye_interest,
                                          properties=('label',
                                                      'area_filled'))
            myelin_of_interest = np.where(label_mye_interest == mye_props['label'][mye_props['area_filled'].argmax()],
                                          True,
                                          False)
            if self.is_struct_open(myelin_of_interest):
                drop_non_closed_myelin.append(mye_index)

        for label_id in ['Axon_seg', 'Myelin_seg']:
            self.filtered_props_df[label_id] = self.filtered_props_df[label_id].drop(drop_non_closed_myelin)
            self.filtered_props_df[label_id] = self.filtered_props_df[label_id].drop_duplicates().reset_index(drop=True)

        centroid_arrays = {'Axon_seg': [], 'Myelin_seg': []}
        for key in centroid_arrays:
            try:
                centroid_arrays[key] = self.filtered_props_df[key][['centroid-0', 'centroid-1']].values
            except KeyError:
                return None

        # Add column for bbox differences between axon and myelin
        self.filtered_props_df['Axon_seg']['Vert_diff'] = None
        self.filtered_props_df['Axon_seg']['Horz_diff'] = None
        # ### Combine for seg
        for axon_idx, axon_cent in enumerate(centroid_arrays['Axon_seg']):
            closest_myelin = cdist([axon_cent], centroid_arrays['Myelin_seg'], 'euclidean').argmin()
            distance = cdist([axon_cent], centroid_arrays['Myelin_seg'], 'euclidean').min()
            if distance < max_com_distance:
                self.final_df['Axon_seg'] = self.final_df['Axon_seg'].append(
                    self.filtered_props_df['Axon_seg'].iloc[axon_idx])
                self.final_df['Myelin_seg'] = self.final_df['Myelin_seg'].append(
                    self.filtered_props_df['Myelin_seg'].iloc[closest_myelin])
                # Store length / width ratio for additional selection criteria
                myel_row = self.final_df['Myelin_seg'].iloc[-1]
                axon_row = self.final_df['Axon_seg'].iloc[-1]
                a_vert = (axon_row['bbox-2'] - axon_row['bbox-0'])
                a_horz = (axon_row['bbox-3'] - axon_row['bbox-1'])
                m_vert = (myel_row['bbox-2'] - myel_row['bbox-0'])
                m_horz = (myel_row['bbox-3'] - myel_row['bbox-1'])
                self.final_df['Axon_seg']['Vert_diff'].iloc[-1] = int(a_vert / m_vert * 100)
                self.final_df['Axon_seg']['Horz_diff'].iloc[-1] = int(a_horz / m_horz * 100)

        myelin_based_outliers = is_outlier(self.final_df['Axon_seg']['Vert_diff'].append(
            self.final_df['Axon_seg']['Horz_diff']))

        # Fill out seg_im_selected
        for idx, label_id in enumerate(['Axon_seg', 'Myelin_seg']):
            self.final_df[label_id] = self.final_df[label_id].reset_index(drop=True)
            self.final_df[label_id] = self.final_df[label_id].drop(myelin_based_outliers)
            self.final_df[label_id] = self.final_df[label_id].drop_duplicates().reset_index(drop=True)

            input_labels = self.props_df[label_id]['label'].values
            output_l = self.final_df[label_id]['label'].values
            output_labels = np.zeros(len(input_labels), dtype=int)
            output_labels[output_l.astype(int) - 1] = output_l
            filtered_image = util.map_array(self.raw_label_img[label_id], input_labels, output_labels)

            filtered_image //= filtered_image
            self.seg_im_selected[self.seg_im_selected == self.axon_myelin_pixel_values[idx]] = 0
            self.seg_im_selected[filtered_image == 1] = self.axon_myelin_pixel_values[idx]

        assert len(self.final_df['Axon_seg']) == len(self.final_df['Myelin_seg']), 'Axon_seg and Myelin_seg lengths unequal'

        # update segmented centroid arrays
        for key in ['Axon_seg', 'Myelin_seg']:
            try:
                centroid_arrays[key] = copy.deepcopy(self.final_df[key][['centroid-0', 'centroid-1']].values)
            except KeyError:
                print('Could not match centroid arrays..')
                return None

        self.final_df['Axon_seg'] = self.final_df['Axon_seg'].reset_index(drop=True)
        self.final_df['Myelin_seg'] = self.final_df['Myelin_seg'].reset_index(drop=True)

        self.final_df['Axon_seg']['Myelin_Thickness'] = abs(self.final_df['Myelin_seg']['equivalent_diameter_area'] -
                                                            self.final_df['Axon_seg']['equivalent_diameter_area']) / 2 *\
                                                        self.pixel_size
        self.final_df['Axon_seg']['Axon_Diameter'] = self.final_df['Axon_seg']['equivalent_diameter_area'] / 2 *\
                                                        self.pixel_size

        return self.final_df, self.seg_im_selected

    def calculate_gratio(self):
        try:
            self.final_df[f'Axon_seg']['AVF'] = self.final_df[f'Axon_seg']['area_filled'] / \
                                                (self.final_df[f'Axon_seg']['area_filled'] +
                                                 self.final_df[f'Myelin_seg']['area_filled'])
            self.final_df[f'Myelin_seg']['MVF'] = self.final_df[f'Myelin_seg']['area_filled'] / \
                                                  (self.final_df[f'Axon_seg']['area_filled'] +
                                                   self.final_df[f'Myelin_seg']['area_filled'])
            self.final_df[f'Axon_seg']['G-ratio'] = np.round(
                np.sqrt(1 / (1 + self.final_df[f'Myelin_seg']['MVF'] /
                             self.final_df[f'Axon_seg']['AVF'])), 3)
        except KeyError:
            print('Could not calculate g-ratio..')
            return None

        return self.final_df

    def is_struct_open(self, binary_image_in):
        binary_image = np.pad(binary_image_in * 255, 2)
        skeleton = cv2.ximgproc.thinning(binary_image.astype('uint8'), None, 1)
        # Threshold the image so that white pixels get a value of 0 and
        # black pixels a value of 10:
        _, binary_image = cv2.threshold(skeleton, 128, 10, cv2.THRESH_BINARY)

        # Set the end-points kernel:
        h = np.array([[1, 1, 1],
                      [1, 10, 1],
                      [1, 1, 1]])

        # Convolve the image with the kernel:
        img_filtered = cv2.filter2D(binary_image, -1, h)

        # Extract only the end-points pixels, those with
        # an intensity value of 110:
        end_points_mask = np.where(img_filtered == 110, 255, 0)

        # The above operation converted the image to 32-bit float,
        # convert back to 8-bit uint
        end_points_mask = end_points_mask.astype(np.uint8)

        return np.sum(end_points_mask) > 1
