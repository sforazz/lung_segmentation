import logging
from pathlib import Path
import nrrd
from basecore.process.preprocess import mouse_lung_data_preparation
from basecore.converters.dicom import DicomConverter
from basecore.process.crop import ImageCropping
from dl.models.unet import mouse_lung_seg
from dl.utils.mouse_segmentation import save_results, preprocessing, postprocessing
from basecore.process.postprocess import binarization
import numpy as np
import os


logger = logging.getLogger('lungs_segmentation')


def run_segmentation(input_dir, work_dir, network_weights, no_crop=False):
    
    input_dir = Path(input_dir)
    logger.info('Input dir: {}'.format(input_dir))
    dcm_folders = sorted([input_dir/x for x in input_dir.iterdir() if x.is_dir() and 
                          ((input_dir/x).glob('*.dcm') or (input_dir/x).glob('*.DCM')
                           or (input_dir/x).glob('*.IMA'))])
    logger.info('Found {} sub folders with DICOM data.'.format(len(dcm_folders)))
    for folder in dcm_folders:
        logger.info('Processing folder {}'.format(folder))
        filename, _, _ = mouse_lung_data_preparation(str(folder), work_dir)
        if filename:
            logger.info('Converting DICOM data to NRRD.')
            converter = DicomConverter(filename, clean=True, bin_path=os.environ['bin_path'])
            converted_data = converter.convert(convert_to='nrrd', method='mitk')
            if not no_crop:
                logger.info('Automatically cropping the nrrd file to have one mouse per image.')
                cropping = ImageCropping(converted_data)
                to_segment = cropping.crop_wo_mask()
                logger.info('Found {} mice in the NRRD file.'.format(len(to_segment)))
            else:
                logger.info('Automatically cropping was disabled. This means the application'
                            ' will assume that there is only one mouse in the image. If this is '
                            'not true, please run the segmentation again without --no-crop.')
                to_segment = [converted_data]
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
            model = mouse_lung_seg()
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
                im = postprocessing(im)
                
                im = binarization(im)

                save_results(im, to_segment[i])
                z = z + s
