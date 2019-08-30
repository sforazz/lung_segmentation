import logging
import os
import numpy as np
from lung_segmentation.crop import ImageCropping
from lung_segmentation.converters.dicom import DicomConverter
from lung_segmentation.generators import load_data_2D
from lung_segmentation.utils import dicom_check, resize_image, split_filename
import nrrd
import glob


logger = logging.getLogger('lungs_segmentation')


class LungSegmentationBase(object):
    
    def __init__(self, input_path, work_dir, deep_check=False):

        self.work_dir = work_dir
        self.deep_check = deep_check
        self.precomputed_masks = []
        self.precomputed_images = []
        self.processed_subs = []
        self.mask_paths = None
        self.input_path = input_path
        self.test_set = []

        if not os.path.isdir(work_dir):
            os.mkdir(work_dir)
    
    def get_data(self):
        raise NotImplementedError('This method has not been implemented yet.')
    
    def preprocessing(self):
        
        self.preprocessed_images = []
        self.preprocessed_masks = []

        if os.path.isfile(os.path.join(self.work_dir, 'processed_DICOM.txt')):
            with open(os.path.join(self.work_dir, 'processed_DICOM.txt'), 'r') as f:
                self.processed_subs = [x.strip() for x in f]
            logger.info('Found {} already processed subjects. They will be skipped '
                        'from the preprocessing.'.format(len(self.processed_subs)))

        if os.path.isfile(os.path.join(self.work_dir, 'processed_NRRD.txt')):
            with open(os.path.join(self.work_dir, 'processed_NRRD.txt'), 'r') as f:
                processed_nrrd = [x.strip() for x in f]
            for sub in processed_nrrd:
                self.precomputed_images = self.precomputed_images + sorted(glob.glob(
                    os.path.join(sub, 'Raw_data*resampled.nrrd')))
                masks = [x for x in sorted(glob.glob(os.path.join(sub, '*resampled.nrrd'))) if 'Raw_data' not in x]
                self.precomputed_masks = self.precomputed_masks + masks
        
        dcm_folders = [x for x in self.dcm_folders if x not in self.processed_subs and x not in self.test_set]
        if self.mask_paths is not None:
            unproceseed_indexes = [i for i, x in enumerate(self.dcm_folders) if x in dcm_folders]
            self.mask_paths = [self.mask_paths[i] for i in unproceseed_indexes]

        self.dcm_folders = dcm_folders

        if self.precomputed_images:
            self.preprocessed_images = self.precomputed_images
        if self.precomputed_masks:
            self.preprocessed_masks = self.precomputed_masks

        self.image_info = {}

        for i, folder in enumerate(self.dcm_folders):
            logger.info('Processing folder {}'.format(folder))
            filename, _, _ = dicom_check(str(folder), self.work_dir,
                                         deep_check=self.deep_check)
            if filename:
                logger.info('Converting DICOM data to NRRD.')
                converter = DicomConverter(filename, clean=True,
                                           bin_path=os.environ['bin_path'])
                converted_data = converter.convert(convert_to='nrrd',
                                                   method='mitk')
                if self.mask_paths is not None:
                    logger.info('Cropping the mouse images based on the '
                                'already segmented lung mask.')
                    images = []
                    masks = []
                    for mask in os.listdir(self.mask_paths[i]):
                        if os.path.isfile(os.path.join(self.mask_paths[i], mask)):
                            prefix = 'Raw_data_for_{}'.format('_'.join(mask.split('.')[:-1]))
                            cropping = ImageCropping(converted_data, 
                                                     os.path.join(self.mask_paths[i], mask),
                                                     prefix=prefix)
                            image, mask = cropping.crop_with_mask()
                            if image is not None and mask is not None:
                                images.append(image)
                                masks.append(mask)
                else:
                    logger.info('Automatically cropping the nrrd file to have'
                                ' one mouse per image (or to remove background in case '
                                'of the original CT has only one mouse already).')
                    prefix = 'Raw_data'
                    cropping = ImageCropping(converted_data, prefix=prefix)
                    images = cropping.crop_wo_mask()
                    masks = []
                logger.info('Found {} mice in the NRRD file.'.format(len(images)))
                for j, image in enumerate(images):
                    logger.info('The cropped images will be now resampled to have 0.5 mm '
                                'isotropic resolution.')
                    _, _, img_path, orig_size = resize_image(image)
                    self.preprocessed_images.append(img_path)
                    self.image_info[img_path] = {}
                    self.image_info[img_path]['orig_size'] = orig_size
                    if masks:
                        _, _, mask_path, _= resize_image(masks[j])
                        self.preprocessed_masks.append(mask_path)
                with open(os.path.join(self.work_dir, 'processed_DICOM.txt'), 'a') as f:
                    f.write(folder+'\n')
                with open(os.path.join(self.work_dir, 'processed_NRRD.txt'), 'a') as f:
                    fname = os.path.split(filename)[0]
                    f.write(fname+'\n')

    def create_tensors(self, patch_size=(96, 96), save2npy=True):
        
        image_tensor = None
        mask_tensor = None
        self.patch_size = patch_size
        for i, image in enumerate(self.preprocessed_images):
            p = 0
            im_base, im_name, _ = split_filename(image)
            im_path = os.path.join(im_base, im_name)
            image, _ = nrrd.read(image)
            im_size = image.shape[:2]
            if self.preprocessed_masks:
                mask = self.preprocessed_masks[i]
                mask_base, mask_name, _ = split_filename(mask)
                mask_path = os.path.join(mask_base, mask_name)
                mask, _ = nrrd.read(mask)
            for z in range(image.shape[2]):
                im_array = load_data_2D('', '', array=image[:, :, z], img_size=im_size,
                                        patch_size=patch_size, binarize=False, normalization=True)
                if self.preprocessed_masks:
                    mask_array = load_data_2D('', '', array=mask[:, :, z], img_size=im_size,
                                              patch_size=patch_size, binarize=True, normalization=False)
                if save2npy:
                    for j in range(im_array.shape[0]):
                        np.save(im_path+'_patch{}.npy'.format(str(p).zfill(5)), im_array)
                        if self.preprocessed_masks:
                            np.save(mask_path+'_patch{}.npy'.format(str(p).zfill(5)), mask_array)
                        p = p+1
                else:
                    if image_tensor is None:
                        image_tensor = im_array
                    else:
                        image_tensor = np.concatenate([image_tensor, im_array], axis=0)
                    if self.preprocessed_masks:
                        if mask_tensor is None:
                            mask_tensor = mask_array
                        else:
                            mask_tensor = np.concatenate([mask_tensor, mask_array], axis=0)

        self.image_tensor = image_tensor
