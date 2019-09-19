"""
Script to run up to the network training.
You can use this script also to run just the pre-processing
 without training the network.
"""
import os
import argparse
from lung_segmentation.utils import create_log
from lung_segmentation.training import LungSegmentationTraining


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--input_file', '-i', type=str,
                        help=('Existing Excel file with a list of all the folders containing DICOMS'
                              ' that will be used for training the network.'))
    PARSER.add_argument('--root_path', '-r', type=str,
                        help=('Path that has to be appended to all the folders in the input_file.'))
    PARSER.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--dcm-check', '-dc', action='store_true',
                        help=('Whether or not to carefully check the DICOM header. '
                              'This check is based on our data and might too stringent for other'
                              ' dataset, so you might want to turn it off. If you turn it off, '
                              'please make sure that the DICOM data are correct. '
                              'Default is False.'))
    PARSER.add_argument('--spacing', '-s', nargs='+', type=int, default=None,
                        help=('New spacing for the resampled images after pre-processing. '
                              'Need to be a list of 3 values. Default is None, so no '
                              'resampling'))
    PARSER.add_argument('--additional-dataset', '-ad', nargs='+', type=str, default=[],
                        help=('Path to the folder with additional dataset that have to be '
                              'for training. They have to be already pre-processed '
                              'using the pre-processing class.'))
    PARSER.add_argument('--epochs', '-e', type=int,
                        help=('Number of epochs. Default is 100.'))
    PARSER.add_argument('--pre-processing-only', '-pp', action='store_true',
                        help=('If True, only the pre-processing will be performed. Useful '
                              'when you want to run the training using more than one dataset. '
                              'You can run the pre-processing first and than the training '
                              'providing the option --additional-dataset.'))
    PARSER.add_argument('--testing', '-ts', action='store_true',
                        help=('If True, some of the folders specified in --input_file '
                              'will be removed from the training and left for testing '
                              'the network performance.'))

    ARGS = PARSER.parse_args()

    PARENT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))
    os.environ['bin_path'] = os.path.join(PARENT_DIR, 'bin/')

    LOG_DIR = os.path.join(PARENT_DIR, 'logs')
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    LOGGER = create_log(LOG_DIR)
    print(ARGS.spacing)
    if ARGS.spacing is not None:
        NEW_SPACING = (ARGS.spacing[0], ARGS.spacing[1], ARGS.spacing[2])
    else:
        NEW_SPACING = None

    WORKFLOW = LungSegmentationTraining(ARGS.input_file, ARGS.work_dir, deep_check=ARGS.dcm_check)
    WORKFLOW.get_data(root_path=ARGS.root_path, testing=ARGS.testing)
    WORKFLOW.preprocessing(new_spacing=NEW_SPACING)
    if not ARGS.pre_processing_only:
        WORKFLOW.create_tensors()
        WORKFLOW.data_split(additional_dataset=ARGS.additional_dataset)
        WORKFLOW.run_training(n_epochs=ARGS.epochs)

# input_dir = '/home/fsforazz/Desktop/PhD_project/fibrosis_project/mouse_list_endo.xlsx'
# work_dir = '/mnt/sdb/endo_no_mask/'
# parent_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))
# 
# os.environ['bin_path'] = os.path.join(parent_dir, 'bin/')
# log_dir = os.path.join(work_dir, 'logs')
# if not os.path.isdir(log_dir):
#     os.makedirs(log_dir)
# 
# logger = create_log(log_dir)
# 
# ls = LungSegmentationTraining(input_dir, work_dir, deep_check=False, tl=False)
# ls.get_data(root_path='/mnt/sdb/mouse_fibrosis_data/')
# ls.preprocessing(new_spacing=(args.spacing[0], 1, 1))
# ls.create_tensors()
# ls.data_split(additional_dataset=['/mnt/sdb/tl_mouse_MA_all/training', '/mnt/sdb/human_seg/training'])
# ls.run_training(n_epochs=100)
