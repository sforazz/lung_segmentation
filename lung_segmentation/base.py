"Class from which all other lung segmentation classes will be derived"
import logging
import os
import glob
import pickle
import nrrd
import numpy as np
from lung_segmentation.crop import ImageCropping
from lung_segmentation.converters.dicom import DicomConverter
from lung_segmentation.generators import load_data_2D
from lung_segmentation.utils import dicom_check, resize_image, split_filename


LOGGER = logging.getLogger('lungs_segmentation')


class LungSegmentationBase():
    "Base class for lung segmentation"
    def __init__(self, input_path, work_dir, deep_check=False, tl=False):

        self.work_dir = work_dir
        self.deep_check = deep_check
        self.precomputed_masks = []
        self.precomputed_images = []
        self.processed_subs = []
        self.mask_paths = None
        self.input_path = input_path
        self.test_set = []
        self.testing = False
        self.preprocessed_images = []
        self.preprocessed_masks = []
        self.dcm_folders = []
        self.image_info = {}
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.image_tensor = None

        if tl:
            LOGGER.info('Trasfer learning training enabled. The encoder part of '
                        'the model will be frozen and only the decoder will be '
                        'trained.')
        self.transfer_learning = tl

        if not os.path.isdir(work_dir):
            os.mkdir(work_dir)

    def get_data(self):
        "Function to get data"
        raise NotImplementedError('This method has not been implemented yet.')

    def preprocessing(self, new_spacing=(0.35, 0.35, 0.35)):
        "Function to pre-process the data"

        if os.path.isfile(os.path.join(self.work_dir, 'processed_DICOM.txt')):
            with open(os.path.join(self.work_dir, 'processed_DICOM.txt'), 'r') as f:
                self.processed_subs = [x.strip() for x in f]
            LOGGER.info('Found {} already processed subjects. They will be skipped '
                        'from the preprocessing.'.format(len(self.processed_subs)))

        if os.path.isfile(os.path.join(self.work_dir, 'processed_NRRD.txt')):
            with open(os.path.join(self.work_dir, 'processed_NRRD.txt'), 'r') as f:
                processed_nrrd = [x.strip() for x in f]
            for sub in processed_nrrd:
                self.precomputed_images = self.precomputed_images + sorted(glob.glob(
                    os.path.join(sub, 'Raw_data*resampled.nrrd')))
                if not self.testing:
                    masks = [x for x in sorted(glob.glob(os.path.join(sub, '*resampled.nrrd')))
                             if 'Raw_data' not in x]
                elif self.testing or new_spacing is None:
                    masks = [x for x in sorted(glob.glob(os.path.join(sub, '*cropped.nrrd')))
                             if 'Raw_data' not in x]
                self.precomputed_masks = self.precomputed_masks + masks
        if os.path.isfile(os.path.join(self.work_dir, 'image_info.p')):
            with open(os.path.join(self.work_dir, 'image_info.p'), 'rb') as fp:
                self.image_info = pickle.load(fp)

        dcm_folders = [x for x in self.dcm_folders if x not in self.processed_subs
                       and x not in self.test_set]
        if self.mask_paths is not None:
            unproceseed_indexes = [i for i, x in enumerate(self.dcm_folders) if x in dcm_folders]
            self.mask_paths = [self.mask_paths[i] for i in unproceseed_indexes]

        self.dcm_folders = dcm_folders

        if self.precomputed_images:
            self.preprocessed_images = self.precomputed_images
        if self.precomputed_masks:
            self.preprocessed_masks = self.precomputed_masks

        for i, folder in enumerate(self.dcm_folders):
            LOGGER.info('Processing folder {}'.format(folder))
            filename, _, _ = dicom_check(str(folder), self.work_dir,
                                         deep_check=self.deep_check)
            if filename:
                LOGGER.info('Converting DICOM data to NRRD.')
                converter = DicomConverter(filename, clean=True,
                                           bin_path=os.environ['bin_path'])
                converted_data = converter.convert(convert_to='nrrd',
                                                   method='mitk')
                if self.mask_paths is not None:
                    LOGGER.info('Cropping the CT images based on the '
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
                    LOGGER.info('Automatically cropping the CT image to have'
                                ' one subject per image (or to remove background in case '
                                'of the original CT has only one subject already).')
                    prefix = 'Raw_data'
                    cropping = ImageCropping(converted_data, prefix=prefix)
                    images = cropping.crop_wo_mask()
                    masks = []
                LOGGER.info('Found {} subjects in the NRRD file.'.format(len(images)))
                for j, image in enumerate(images):
                    if new_spacing is not None:
                        LOGGER.info('The cropped images will be now resampled to have '
                                    '{0}x{1}x{2} mm resolution.'
                                    .format(new_spacing[0], new_spacing[1], new_spacing[2]))
                        _, _, img_path, orig_size = resize_image(image, new_spacing=new_spacing)
                    else:
                        img_path = image
                        orig_size = None
                    self.preprocessed_images.append(img_path)
                    self.image_info[img_path] = {}
                    self.image_info[img_path]['orig_size'] = orig_size
                    if masks and not self.testing:
                        if new_spacing is not None:
                            _, _, mask_path, _= resize_image(masks[j], new_spacing=new_spacing)
                        else:
                            mask_path = masks[j]
                        self.preprocessed_masks.append(mask_path)
                    elif masks and self.testing:
                        self.preprocessed_masks.append(masks[j])
                with open(os.path.join(self.work_dir, 'processed_DICOM.txt'), 'a') as f:
                    f.write(str(folder)+'\n')
                with open(os.path.join(self.work_dir, 'processed_NRRD.txt'), 'a') as f:
                    fname = os.path.split(filename)[0]
                    f.write(fname+'\n')

        with open(os.path.join(self.work_dir, 'image_info.p'), 'wb') as fp:
            pickle.dump(self.image_info, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def create_tensors(self, patch_size=(96, 96), save2npy=True):
        "Function to create the 2D tensor from the 3D images"
        image_tensor = []
        LOGGER.info('Creating the patches to fed then into the network.')
        LOGGER.info('Chosen path size is: {0}x{1}.'.format(patch_size[0], patch_size[1]))
        for i, image in enumerate(self.preprocessed_images):
            patch = 0
            im_base, im_name, ext = split_filename(image)
            im_path = os.path.join(im_base, im_name)
            if not glob.glob(im_path+'*.npy'):
                image, _ = nrrd.read(image)
                im_size = image.shape[:2]
                if self.preprocessed_masks and not self.testing:
                    mask = self.preprocessed_masks[i]
                    mask_base, mask_name, _ = split_filename(mask)
                    mask_path = os.path.join(mask_base, mask_name)
                    mask, _ = nrrd.read(mask)
                for n_slice in range(image.shape[2]):
                    im_array, info_dict = load_data_2D(
                        '', '', array=image[:, :, n_slice], img_size=im_size,
                        patch_size=patch_size, binarize=False, normalization=True,
                        prediction=self.testing)
                    if self.preprocessed_masks and not self.testing:
                        mask_array, _ = load_data_2D(
                            '', '', array=mask[:, :, n_slice], img_size=im_size,
                            patch_size=patch_size, binarize=True, normalization=False)
                    if save2npy:
                        for j in range(im_array.shape[0]):
                            np.save(im_path+'_patch{}.npy'
                                    .format(str(patch).zfill(5)), im_array[j, :])
                            if self.preprocessed_masks:
                                np.save(mask_path+'_patch{}.npy'
                                        .format(str(patch).zfill(5)), mask_array[j, :])
                            patch = patch+1
                    else:
                        for j in range(im_array.shape[0]):
                            image_tensor.append(im_array[j, :])
                if not save2npy:
                    if info_dict is not None:
                        im_name = im_path+ext
                        self.image_info[im_name]['slices'] = n_slice+1
                        for k in info_dict[0].keys():
                            self.image_info[im_name][k] = info_dict[0][k]
        if image_tensor:
            self.image_tensor = (np.asarray(image_tensor).reshape(
                -1, im_array.shape[1], im_array.shape[2], 1))
