"Simple script to run segmentation inference"
import os
import argparse
import glob
from lung_segmentation.utils import create_log, untar, get_files
from lung_segmentation.inference import LungSegmentationInference
from lung_segmentation.configuration import (
    STANDARD_CONFIG, HIGH_RES_CONFIG, HUMAN_CONFIG)


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--input_path', '-i', type=str,
                        help=('Existing Excel file with a list of all the folders containing DICOMS'
                              ' that will be used for training the network.'))
    PARSER.add_argument('--root_path', '-r', type=str, default='',
                        help=('Path that has to be appended to all the folders in the input_file.'))
    PARSER.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--configuration', '-c', type=str,
                        choices=['standard', 'highres', 'human', None], default='standard',
                        help=('Configuation to use based on your data. See documentation for '
                              'more help. Default is "standard".'))
    PARSER.add_argument('--min-extent', type=int, default=350,
                        help=('Minimum lung extension (in voxels) that will be used to '
                              'run the final correction after the inference. For mouse acquired '
                              'with clinical CT, 350 should be enough, while for micro-CT or human '
                              'date this should be set to 30000-40000. Default is 350.'))
    PARSER.add_argument('--dcm-check', '-dc', action='store_true',
                        help=('Whether or not to carefully check the DICOM header. '
                              'This check is based on our data and might too stringent for other'
                              ' dataset, so you might want to turn it off. If you turn it off, '
                              'please make sure that the DICOM data are correct. '
                              'Default is False.'))
    PARSER.add_argument('--spacing', '-s', nargs='+', type=float, default=None,
                        help=('New spacing for the resampled images after pre-processing. '
                              'Need to be a list of 3 values. Default is None, so no '
                              'resampling'))
    PARSER.add_argument('--weights', nargs='+', type=str, default=None,
                        help=('Path to the CNN weights to be used for the inference '
                              ' More than one weight can be used, in that case the average '
                              'prediction will be returned.'))
    PARSER.add_argument('--cluster-correction', '-cc', action='store_true',
                        help=('Whether or not to apply cluster correction to the final segmented '
                              'image. This should be turned on when segmenting human or high res '
                              'mouse data. If not provided, the segmented mask will be thresholded '
                              'based on the Otsu threshold. If provided, take also a look to the '
                              '--min-extent argument since it is used to choose the cluster '
                              'dimension. Default is False.'))
    PARSER.add_argument('--evaluate', action='store_true',
                        help=('If ground truth lung masks are available, the result of the '
                              'segmentation can be tested against them. In this case, both '
                              'Dice score and Hausdorff distance will be calculated. '
                              'Default is False.'))

    ARGS = PARSER.parse_args()

    PARENT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))
    BIN_DIR = os.path.join(PARENT_DIR, 'bin/')
    WEIGHTS_DIR = os.path.join(PARENT_DIR, 'weights/')
    BIN_URL = 'https://angiogenesis.dkfz.de/oncoexpress/software/delineation/bin/bin.tar.gz'

    if ARGS.configuration is None:
        DEEP_CHECK = ARGS.dcm_check
        NEW_SPACING = ARGS.spacing
        MIN_EXTENT = ARGS.min_extent
        CLUSTER_CORRECTION = ARGS.cluster_correction
        WEIGHTS_URL = None
    else:
        if ARGS.configuration == 'standard':
            CONFIG = STANDARD_CONFIG
        elif ARGS.configuration == 'highres':
            CONFIG = HIGH_RES_CONFIG
        elif ARGS.configuration == 'human':
            CONFIG = HUMAN_CONFIG

        DEEP_CHECK = CONFIG['dicom_check']
        NEW_SPACING = CONFIG['spacing']
        MIN_EXTENT = CONFIG['min_extent']
        CLUSTER_CORRECTION = CONFIG['cluster_correction']
        WEIGHTS_URL = CONFIG['weights_url']

    os.environ['bin_path'] = BIN_DIR

    LOG_DIR = os.path.join(ARGS.work_dir, 'logs')
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOGGER = create_log(LOG_DIR)

    if ARGS.weights is None and WEIGHTS_URL is not None:
        if not os.path.isdir(WEIGHTS_DIR):
            LOGGER.info('No pre-trained network weights, I will try to download them.')
            try:
                TAR_FILE = get_files(WEIGHTS_URL, PARENT_DIR, 'weights')
                untar(TAR_FILE)
            except:
                LOGGER.error('Pre-trained weights cannot be downloaded. Please check '
                             'your connection and retry or download them manually '
                             'from the repository.')
                raise Exception('Unable to download network weights!')
        else:
            LOGGER.info('Pre-trained network weights found in %s', WEIGHTS_DIR)

        WEIGHTS = [w for w in sorted(glob.glob(os.path.join(WEIGHTS_DIR, '*.h5')))]
        DOWNLOADED = True
    elif ARGS.weights is not None:
        WEIGHTS = ARGS.weights
        DOWNLOADED = False
    else:
        LOGGER.error('If you choose to do not use any configuration file, '
                     'then you must provide the path to the weights to use for '
                     'inference!')
        raise Exception('No weights can be found')
    if len(WEIGHTS) == 5 and DOWNLOADED:
        LOGGER.info('%s weights files found in %s. Five folds inference will be calculated.',
                    len(WEIGHTS), WEIGHTS_DIR)
    elif WEIGHTS and len(WEIGHTS) < 5 and DOWNLOADED:
        LOGGER.warning('Only %s weights files found in %s. There should be 5. Please check '
                       'the repository and download them again in order to run the five folds '
                       'inference will be calculated. The segmentation will still be calculated '
                       'using %s-folds cross validation but the results might be sub-optimal.',
                       len(WEIGHTS), WEIGHTS_DIR, len(WEIGHTS))
    elif len(WEIGHTS) > 5 and DOWNLOADED:
        LOGGER.error('%s weights file found in %s. This is not possible since the model was '
                     'trained using a 5-folds cross validation approach. Please check the '
                     'repository and remove all the unknown weights files.',
                     len(WEIGHTS), WEIGHTS_DIR)
    elif not WEIGHTS:
        LOGGER.error('No weights file found in %s. Probably something went wrong '
                     'during the download. Try to download them directly from %s '
                     'and store them in the "weights" folder within the '
                     'lung_segmentation directory.', WEIGHTS_DIR, WEIGHTS_URL)
        raise Exception('No weight files found!')

    if not os.path.isdir(BIN_DIR):
        LOGGER.info('No directory containing the binary executables found. '
                    'I will try to download it from the repository.')
        try:
            TAR_FILE = get_files(BIN_URL, PARENT_DIR, 'bin')
            untar(TAR_FILE)
        except:
            LOGGER.error('Binary files cannot be downloaded. Please check '
                         'your connection and retry or download them manually '
                         'from the repository.')
            raise Exception('Unable to download binary files!')
    else:
        LOGGER.info('Binary executables found in %s', BIN_DIR)

    LOGGER.info('The following configuration will be used for the inference:')
    LOGGER.info('Input path: %s', ARGS.input_path)
    LOGGER.info('Working directory: %s', ARGS.work_dir)
    LOGGER.info('Root path: %s', ARGS.root_path)
    LOGGER.info('DICOM check: %s', DEEP_CHECK)
    LOGGER.info('New spacing: %s', NEW_SPACING)
    LOGGER.info('Weight files: \n%s', '\n'.join([x for x in sorted(WEIGHTS)]))
    LOGGER.info('Cluster correction: %s', CLUSTER_CORRECTION)
    if CLUSTER_CORRECTION:
        LOGGER.info('Minimum extent: %s', MIN_EXTENT)
    LOGGER.info('Evaluation: %s', ARGS.evaluate)

    INFERENCE = LungSegmentationInference(ARGS.input_path, ARGS.work_dir, deep_check=DEEP_CHECK)
    INFERENCE.get_data(root_path=ARGS.root_path)
    INFERENCE.preprocessing(new_spacing=NEW_SPACING)
    INFERENCE.create_tensors()
    INFERENCE.run_inference(weights=WEIGHTS)
    INFERENCE.save_inference(min_extent=MIN_EXTENT, cluster_correction=CLUSTER_CORRECTION)
    if ARGS.evaluate:
        INFERENCE.run_evaluation()

print('Done!')
