import logging
from pathlib import Path
import nrrd
from lung_segmentation.utils import dicom_check
from lung_segmentation.converters.dicom import DicomConverter
from lung_segmentation.crop import ImageCropping
from lung_segmentation.models import unet_lung
from lung_segmentation.utils import save_results, preprocessing, postprocessing
from lung_segmentation.utils import binarization
import numpy as np
import os


logger = logging.getLogger('lungs_segmentation')


def pipeline(input_dir, work_dir, network_weights, deep_check=False):
    
    input_dir = Path(input_dir)
    logger.info('Input dir: {}'.format(input_dir))
    dcm_folders = sorted([input_dir/x for x in input_dir.iterdir() if x.is_dir() and 
                          ((input_dir/x).glob('*.dcm') or (input_dir/x).glob('*.DCM')
                           or (input_dir/x).glob('*.IMA'))])
    logger.info('Found {} sub folders with DICOM data.'.format(len(dcm_folders)))
    for folder in dcm_folders:
        logger.info('Processing folder {}'.format(folder))
#         if not deep_check:
        filename, _, _ = dicom_check(str(folder), work_dir, deep_check=deep_check)
#         else:
#             folder_name = folder.split('/')[-1]
#             filename = str(list(folder.glob('*'))[0])
        if filename:
            logger.info('Converting DICOM data to NRRD.')
            converter = DicomConverter(filename, clean=True, bin_path=os.environ['bin_path'])
            converted_data = converter.convert(convert_to='nrrd', method='mitk')
            logger.info('Automatically cropping the nrrd file to have one mouse per image '
                        '(or to remove background in case of the original CT has only one mouse already).')
            cropping = ImageCropping(converted_data)
            to_segment = cropping.crop_wo_mask()
            to_segment = [converted_data]
            logger.info('Found {} mice in the NRRD file.'.format(len(to_segment)))
            test_set = []
            n_slices = []
            logger.info('Pre-processing the data before feeding the images into the segmentation network.')
            for im in to_segment:
                im, _ = nrrd.read(im)
                n_slices.append(im.shape[2])
                for s in range(im.shape[2]):
                    sl = preprocessing(im[:, :, s])
                    test_set.append(sl)
            
            test_set = np.asarray(test_set)
            predictions = []
            logger.info('Segmentation inference started.')
            model = unet_lung()
            for i, w in enumerate(network_weights):
                logger.info('Segmentation inference fold {}.'.format(i+1))
                model.load_weights(w)
                predictions.append(model.predict(test_set))
                
            predictions = np.asarray(predictions, dtype=np.float32)
            prediction = np.mean(predictions, axis=0)
            
            z = 0
            logger.info('Binarizing and saving the results.')
            for i, s in enumerate(n_slices):
                im = prediction[z:z+s, :, :, 0]
                im = postprocessing(im, method='human')
                
                im = binarization(im)

                save_results(im, to_segment[i])
                z = z + s
