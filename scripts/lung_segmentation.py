from pathlib import Path
import logging
import nrrd
from basecore.process.preprocess import mouse_lung_data_preparation
from basecore.converters.dicom import DicomConverter
from basecore.process.crop import ImageCropping
from dl.models.unet import mouse_lung_seg
from dl.utils.mouse_segmentation import save_results, preprocessing, postprocessing
from basecore.process.postprocess import binarization
import numpy as np
import time
import glob
import os
from datetime import datetime


now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
logger = logging.getLogger('lungs_segmentation')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(
    os.path.split(__file__)[0],'lungs_segmentation_{}.log'.format(dt_string)))
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


input_dir = '/mnt/sdb/test_lung_seg/'
work_dir = '/mnt/sdb/test_lung_seg_out/'


def data_preparation(input_dir, work_dir, network_weights):
    
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
            converter = DicomConverter(filename, clean=True)
            converted_data = converter.convert(convert_to='nrrd', method='mitk')
            logger.info('Automatically cropping the nrrd file to have one mouse per image.')
            cropping = ImageCropping(converted_data)
            to_segment = cropping.crop_wo_mask()
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

weights = [w for w in sorted(glob.glob(os.path.join(os.path.split(__file__)[0], 'weights/*.h5')))]
# weights = ['/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_1.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_2.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_3.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_4.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_5.h5']

logger.info('Segmentation started!')
start = time.perf_counter()

data_preparation(input_dir, work_dir, weights)

stop = time.perf_counter()
logger.info('Segmentation ended after {} seconds!'.format(int(stop-start)))
