#!/usr/bin/env python3.6
import os
import glob
from lung_segmentation.utils import create_log, untar, get_files, build_gui
from lung_segmentation.inference import LungSegmentationInference


VALUES = build_gui()

PARENT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))
BIN_DIR = os.path.join(PARENT_DIR, 'bin/')
WEIGHTS_DIR = os.path.join(PARENT_DIR, 'weights/')
BIN_URL = 'https://angiogenesis.dkfz.de/oncoexpress/software/delineation/bin/bin.tar.gz'

WORK_DIR = VALUES['work_dir']
INPUT_PATH = VALUES['in_path']
ROOT_PATH = VALUES['root_path']
NEW_SPACING = (VALUES['space_x'], VALUES['space_y'], VALUES['space_z'])
MIN_EXTENT = VALUES['min_extent']
CLUSTER_CORRECTION = VALUES['cluster_correction']
DICOM_CHECK = VALUES['dcm_check']
EVALUATE = VALUES['evaluate']
WEIGHTS_URL = VALUES['weights_url']

try:
    weights_dir = VALUES['weights_dir']
    weights = [w for w in sorted(glob.glob(os.path.join(weights_dir, '*.h5')))]
    if not weights:
        weights = None
except KeyError:
    weights = None

os.environ['bin_path'] = BIN_DIR

LOG_DIR = os.path.join(WORK_DIR, 'logs')
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)

LOGGER = create_log(LOG_DIR)

if weights is None and WEIGHTS_URL is not None:
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
elif weights is not None:
    WEIGHTS = weights
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
LOGGER.info('Input path: %s', INPUT_PATH)
LOGGER.info('Working directory: %s', WORK_DIR)
LOGGER.info('Root path: %s', ROOT_PATH)
LOGGER.info('DICOM check: %s', DICOM_CHECK)
LOGGER.info('New spacing: %s', NEW_SPACING)
LOGGER.info('Weight files: \n%s', '\n'.join([x for x in sorted(WEIGHTS)]))
LOGGER.info('Cluster correction: %s', CLUSTER_CORRECTION)
if CLUSTER_CORRECTION:
    LOGGER.info('Minimum extent: %s', MIN_EXTENT)
LOGGER.info('Evaluation: %s', EVALUATE)

INFERENCE = LungSegmentationInference(INPUT_PATH, WORK_DIR, deep_check=DICOM_CHECK)
INFERENCE.get_data(root_path=ROOT_PATH)
INFERENCE.preprocessing(new_spacing=NEW_SPACING)
INFERENCE.create_tensors()
INFERENCE.run_inference(weights=WEIGHTS)
INFERENCE.save_inference(min_extent=MIN_EXTENT, cluster_correction=CLUSTER_CORRECTION)
if EVALUATE:
    INFERENCE.run_evaluation()

print('Done!')
