"Class to run lung segmentation inference"
import logging
import os
from pathlib import Path
import nrrd
import numpy as np
from skimage.transform import resize
from lung_segmentation.utils import (binarization, dice_calculation,
                                     violin_box_plot, cluster_correction,
    eucl_max)
from lung_segmentation.models import unet_lung
from lung_segmentation.base import LungSegmentationBase


LOGGER = logging.getLogger('lungs_segmentation')


class LungSegmentationInference(LungSegmentationBase):
    "Class to run the lung segmentation inference and evaluation."
    def get_data(self):
        "Function to get the data for the prediction"
        self.testing = True
        self.predicted_images = []
        if (os.path.isdir(os.path.join(self.work_dir, 'testing'))
                and os.path.isfile(os.path.join(self.work_dir, 'testing', 'test_subjects.txt'))):
            with open(os.path.join(self.work_dir, 'testing', 'test_subjects.txt'), 'r') as f:
                self.dcm_folders = [x.strip() for x in f]
        if (os.path.isdir(os.path.join(self.work_dir, 'testing'))
                and os.path.isfile(os.path.join(self.work_dir, 'testing',
                                                'test_subjects_gt_masks.txt'))):
            with open(os.path.join(self.work_dir, 'testing',
                                   'test_subjects_gt_masks.txt'), 'r') as f:
                self.mask_paths = [x.strip() for x in f]
        else:
            LOGGER.info('No folder called "testing" in the working directory.'
                        ' The pipeline will look for DICOM file to use for '
                        'inference in all the sub-folders within the '
                        'working directory.')
            input_dir = Path(self.input_path)
            LOGGER.info('Input directory: {}'.format(input_dir))
            self.dcm_folders = sorted([input_dir/x for x in input_dir.iterdir() if x.is_dir() and
                                      ((input_dir/x).glob('*.dcm') or (input_dir/x).glob('*.DCM')
                                       or (input_dir/x).glob('*.IMA'))])
            LOGGER.info('Found {0} sub-folders in {1}. They will be used to run the inference.'
                        .format(len(self.dcm_folders), str(input_dir)))

        self.work_dir = os.path.join(str(self.work_dir), 'testing')

    def create_tensors(self, patch_size=(96, 96), save2npy=False):
        "Function to create the tensors for the prediction"
        return LungSegmentationBase.create_tensors(self, patch_size=patch_size, save2npy=save2npy)

    def run_inference(self, weights):
        "Function to run the CNN inference"
        test_set = np.asarray(self.image_tensor)
        predictions = []
        LOGGER.info('Segmentation inference started.')
        model = unet_lung()
        for i, weight in enumerate(weights):
            LOGGER.info('Segmentation inference fold {}.'.format(i+1))
            model.load_weights(weight)
            predictions.append(model.predict(test_set))

        predictions = np.asarray(predictions, dtype=np.float32)
        self.prediction = np.mean(predictions, axis=0)

    def save_inference(self):
        "Function to save the segmented masks"
        prediction = self.prediction
        z0 = 0
        for i, image in enumerate(self.image_info):
            patches = self.image_info[image]['patches']
            slices = self.image_info[image]['slices']
            resampled_image_dim = self.image_info[image]['image_dim']
            indexes = self.image_info[image]['indexes']
            deltas = self.image_info[image]['deltas']
            original_image_dim = self.image_info[image]['orig_size']
            im = prediction[z0:z0+(slices*patches), :, :, 0]
            final_prediction = self.inference_reshaping(
                im, patches, slices, resampled_image_dim, indexes, deltas,
                original_image_dim, binarize=False)
            outname = image.split('_resampled')[0]+'_lung_segmented.nrrd'
            reference = image.split('_resampled')[0]+'.nrrd'
            _, hd = nrrd.read(reference)
            nrrd.write(outname, final_prediction, header=hd)
            outname = cluster_correction(outname, 0.2, 10000)
            self.predicted_images.append(outname)
            z0 = z0+(slices*patches)

    @staticmethod
    def inference_reshaping(generated_images, patches, slices,
                            dims, indexes, deltas, original_size,
                            binarize=False):
        "Function to reshape the predictions"
        if patches > 1:
            sl = 0
            final_image = np.zeros((slices, dims[0], dims[1], patches),
                                   dtype=np.float32)-2
            for n in range(0, generated_images.shape[0], patches):
                k = 0
                for j in indexes[1]:
                    for i in indexes[0]:
                        final_image[sl, i[0]:i[1], j[0]:j[1], k] = (
                            generated_images[n+k, deltas[0]:, deltas[1]:])
                        k += 1
                sl = sl + 1
            final_image[final_image==-2] = np.nan
            final_image = np.nanmean(final_image, axis=-1)
            final_image[np.isnan(final_image)] = 0
        else:
            final_image = generated_images[:, deltas[0]:, deltas[1]:]

        final_image = np.swapaxes(final_image, 0, 2)
        final_image = np.swapaxes(final_image, 0, 1)
        if final_image.shape != original_size:
            final_image = resize(final_image.astype(np.float64), original_size, order=0,
                                 mode='edge', cval=0, anti_aliasing=False)
        if binarize:
            final_image = binarization(final_image)

        return final_image

    def run_evaluation(self):
        "Function to evaluate the segmentation w.r.t. a ground truth"
        assert len(self.predicted_images) == len(self.preprocessed_masks)
        all_dsc = []
        all_hd = []
        for i, predicted in enumerate(self.predicted_images):
            gt = self.preprocessed_masks[i]
            dsc = dice_calculation(gt, predicted)
            all_dsc.append(dsc)
            hd_95 = eucl_max(gt, predicted, new_spacing=(1, 1, 1))
            all_hd.append(hd_95)

        violin_box_plot(all_dsc, os.path.join(self.work_dir, 'DSC_violin_plot.png'))
        violin_box_plot(all_hd, os.path.join(self.work_dir, 'HD_violin_plot.png'))

        LOGGER.info('Median DSC: {}'.format(np.median(all_dsc)))
        LOGGER.info('Standard Deviation DSC: {}'.format(np.std(all_dsc)))
        LOGGER.info('Median HD: {0}'.format(np.median(all_hd)))
        LOGGER.info('Standard Deviation HD: {}'.format(np.std(all_hd)))
        LOGGER.info('Max DSC: {}'.format(np.max(all_dsc)))
        LOGGER.info('Min DSC: {}'.format(np.min(all_dsc)))
        LOGGER.info('Max HD: {}'.format(np.max(all_hd)))
        LOGGER.info('Min HD: {}'.format(np.min(all_hd)))
