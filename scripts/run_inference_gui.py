#!/usr/bin/env python3.6
import os
import glob
import PySimpleGUI as sg
from lung_segmentation.utils import create_log, untar, get_files
from lung_segmentation.inference import LungSegmentationInference
from lung_segmentation.configuration import (
    STANDARD_CONFIG, HIGH_RES_CONFIG, HUMAN_CONFIG, NONE_CONFIG)

sg.ChangeLookAndFeel('GreenTan')

layout = [
    [sg.Text('Lung Segmentation using CNN', size=(30, 1), font=("Helvetica", 25))],
    [sg.Listbox(values=('Low resolution mouse', 'High resolution mouse', 'Human', 'None'),
                size=(30, 3), default_values='Low resolution mouse')],
    [sg.Submit(), sg.Cancel()]
]

window = sg.Window('SIENA', default_element_size=(40, 1)).Layout(layout)
button, values = window.Read()

if values[0][0] == 'Low resolution mouse':
    CONFIG = STANDARD_CONFIG
    CC_RANGE = (100, 1000)
    APP_NAME = 'Lung segmentation inference for low resolution mouse CT images'
elif values[0][0] == 'High resolution mouse':
    CONFIG = HIGH_RES_CONFIG
    CC_RANGE = (50000, 500000)
    APP_NAME = 'Lung segmentation inference for high resolution mouse CT images'
elif values[0][0] == 'Human':
    CONFIG = HUMAN_CONFIG
    CC_RANGE = (50000, 500000)
    APP_NAME = 'Lung segmentation inference for human CT images'
else:
    CONFIG = NONE_CONFIG
    CC_RANGE = (0, 500000)
    APP_NAME = 'Lung segmentation inference using custom settings'

DICOM_CHECK = CONFIG['dicom_check']
NEW_SPACING = CONFIG['spacing']
MIN_EXTENT = CONFIG['min_extent']
CLUSTER_CORRECTION = CONFIG['cluster_correction']
WEIGHTS_URL = CONFIG['weights_url']

post_proc_layout = [
    [sg.Radio('Cluster correction', 'pp', default=CLUSTER_CORRECTION, key='cluster_correction'),
     sg.InputText(('Minimum extent'), size=(20, 3)),
     sg.Slider(range=CC_RANGE, orientation='h', size=(34, 20), default_value=MIN_EXTENT,
               key='min_extent')],
    [sg.Radio('Otsu binarizazion', 'pp', default=not(CLUSTER_CORRECTION))],
    [sg.Checkbox('Run segmentation evaluation', default=False,
                 tooltip='',
                 key='evaluate')]]
input_layout = [
    [sg.Text('Input path', size=(15, 1),
             auto_size_text=False, justification='right'),
     sg.InputText('', key='in_path'), sg.FolderBrowse()],
    [sg.Text('Root path', size=(15, 1), auto_size_text=False, justification='right'),
     sg.InputText('', key='root_path'), sg.FolderBrowse()],
    [sg.Text('Working directory', size=(15, 1), auto_size_text=False, justification='right'),
     sg.InputText('', key='work_dir'), sg.FolderBrowse()]
    ]
if WEIGHTS_URL is None:
    input_layout.append(
        [sg.Text('Weights directory', size=(15, 1), auto_size_text=False,
                 justification='right'),
         sg.InputText('', key='weights_dir'), sg.FolderBrowse()])

colx = [[sg.Text('Spacing X')],
        [sg.Slider(range=(0.0, 5.0), orientation='v', resolution=0.05, size=(5, 20),
                   default_value=NEW_SPACING[0],
                   tooltip='New resolution (in mm) along x direction.',
                   key='space_x')]]
coly = [[sg.Text('Spacing Y')],
        [sg.Slider(range=(0.0, 5.0), orientation='v', resolution=0.05, size=(5, 20),
                   default_value=NEW_SPACING[1],
                   tooltip='New resolution (in mm) along y direction.',
                   key='space_y')]]
colz = [[sg.Text('Spacing Z')],
        [sg.Slider(range=(0.0, 5.0), orientation='v', resolution=0.05, size=(5, 20),
                   default_value=NEW_SPACING[2],
                   tooltip='New resolution (in mm) along z direction.',
                   key='space_z')]]
preproc_layout = [
    [sg.Checkbox('DICOM check', default=DICOM_CHECK,
                 tooltip='Whether or not to carefully check the DICOM header. \n'
                         'This check is based on our low resolution mouse data \n'
                         'and might be too stringent for data coming from \n'
                         'different sites and/or acquired with different resolution.\n'
                         'If the last is the case, then turn this check off.',
                 key='dcm_check')],
    [sg.Text('New spacing (for resampling)',
             tooltip='Here you can specify the voxel size (in mm) \n'
                     'that will be used to resample the image before\n'
                     'running the inference. The network has been trained\n'
                     'using images with a defined resolution. The default\n'
                     'values for the resolution have been stored here, \n'
                     'however, if the network was retrained with different\n'
                     'resolution, you can simply change it here.')],
    [sg.Column(colx), sg.Column(coly), sg.Column(colz)]]

layout = [
    [sg.Text(APP_NAME, size=(50, 1),
             font=("Helvetica", 25))],
    [sg.Frame('Input and output specs', input_layout, font='Any 12', title_color='black')],
    [sg.Text('_'  * 80)],
    [sg.Frame('Pre-processing specs', preproc_layout, font='Any 12', title_color='black')],
    [sg.Text('_'  * 80)],
    [sg.Frame('Post-processing specs', post_proc_layout, font='Any 12', title_color='black')],
    [sg.Submit(), sg.Cancel()]]
window = sg.Window('SIENA2', default_element_size=(50, 1)).Layout(layout)
button, VALUES = window.Read()
# sg.Popup(button, values)

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

try:
    weights_dir = VALUES['weights_dir']
    weights = [w for w in sorted(glob.glob(os.path.join(weights_dir, '*.h5')))]
    if not weights:
        weights = None
except:
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
