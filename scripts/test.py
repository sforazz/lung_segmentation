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
    PARSER.add_argument('--root_path', '-r', type=str, default='',
                        help=('Path that has to be appended to all the folders in the input_file.'))
    PARSER.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
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
    PARSER.add_argument('--additional-dataset', '-ad', nargs='+', type=str, default=[],
                        help=('Path to the folder with additional dataset that have to be '
                              'for training. They have to be already pre-processed '
                              'using the pre-processing class.'))
    PARSER.add_argument('--epochs', '-e', type=int, default=100,
                        help=('Number of epochs. Default is 100.'))
    PARSER.add_argument('--pre-processing-only', '-pp', action='store_true',
                        help=('If True, only the pre-processing will be performed. Useful '
                              'when you want to run the training using more than one dataset. '
                              'You can run the pre-processing first and than the training '
                              'providing the option --additional-dataset.'))
    PARSER.add_argument('--create-tensors', action='store_true',
                        help=('If True and --pre-processing-only was selected, the tensors '
                              'for the training will be created.'))
    PARSER.add_argument('--testing', '-t', action='store_true',
                        help=('If True, some of the folders specified in --input_file '
                              'will be removed from the training and left for testing '
                              'the network performance.'))
    PARSER.add_argument('--keep-training', action='store_true',
                        help=('If True, the training will continue from the latest '
                              'saved timepoint.'))
    PARSER.add_argument('--transfer-learning', '-tl', action='store_true',
                        help=('If True, a transfer learning approach will be used to '
                              'train on a new dataset. In this case, only the decoder '
                              'part of the UNet will be trained.'))
    PARSER.add_argument('--pretrained-weights', '-pw', type=str,
                        help=('If --keep-training or --transfer-learning, then you have '
                              'to specify the path to the pre-trained weights using this '
                              'command.'))
    PARSER.add_argument('--use-data-augmentation', '-da', action='store_true',
                        help=('If True, data augmentation will be used during training.'))
    PARSER.add_argument('--training-steps', '-ts', type=int, default=None,
                        help=('Number of training steps per epoch. Default is training size '
                              'divided by training batch size.'))
    PARSER.add_argument('--validation-steps', '-vs', type=int, default=None,
                        help=('Number of validation steps per epoch. Default is validation size '
                              'divided by validation batch size.'))

    ARGS = PARSER.parse_args()

    PARENT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))
    os.environ['bin_path'] = os.path.join(PARENT_DIR, 'bin/')

    LOG_DIR = os.path.join(ARGS.work_dir, 'logs')
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    LOGGER = create_log(LOG_DIR)

    if ARGS.spacing is not None:
        NEW_SPACING = (ARGS.spacing[0], ARGS.spacing[1], ARGS.spacing[2])
    else:
        NEW_SPACING = None

    WORKFLOW = LungSegmentationTraining(ARGS.input_file, ARGS.work_dir,
                                        deep_check=ARGS.dcm_check,
                                        tl=ARGS.transfer_learning)
    WORKFLOW.get_data(root_path=ARGS.root_path, testing=ARGS.testing)
    WORKFLOW.preprocessing(new_spacing=NEW_SPACING)
    if not ARGS.pre_processing_only:
        WORKFLOW.create_tensors()
        WORKFLOW.data_split(additional_dataset=ARGS.additional_dataset)
#         WORKFLOW.prepare_training_parameters(
#             n_epochs=ARGS.epochs, keep_training=ARGS.keep_training,
#             weight_name=ARGS.pretrained_weights, training_steps=ARGS.training_steps,
#             validation_steps=ARGS.validation_steps)
        WORKFLOW.run_training_augmented(
            n_epochs=ARGS.epochs, keep_training=ARGS.keep_training,
            weight_name=ARGS.pretrained_weights, training_steps=ARGS.training_steps,
            validation_steps=ARGS.validation_steps,
            data_augmentation=ARGS.use_data_augmentation)
#         else:
#             WORKFLOW.run_training()
    elif ARGS.pre_processing_only and ARGS.create_tensors:
        WORKFLOW.create_tensors()

LOGGER.info('Everything is done!')
