"Simple script to run individual segmentation inference "
"on already cropped mouse images, in NRRD format"
import os
import argparse
from lung_segmentation.utils import create_log
from lung_segmentation.inference import IndividualInference


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--input_path', '-i', type=str,
                        help=('Existing Excel file with a list of all the folders containing DICOMS'
                              ' that will be used for training the network.'))
    PARSER.add_argument('--root_path', '-r', type=str, default='',
                        help=('Path that has to be appended to all the folders in the input_file.'))
    PARSER.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--min-extent', type=int, default=350,
                        help=('Minimum lung extension (in voxels) that will be used to '
                              'run the final correction after the inference. For mouse acquired '
                              'with clinical CT, 350 should be enough, while for micro-CT or human '
                              'data this should be set to 100000-300000. Default is 350.'))
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

    os.environ['bin_path'] = BIN_DIR

    LOG_DIR = os.path.join(ARGS.work_dir, 'logs')
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOGGER = create_log(LOG_DIR)

    WEIGHTS = ARGS.weights
    DOWNLOADED = False

    LOGGER.info('The following configuration will be used for the inference:')
    LOGGER.info('Input path: %s', ARGS.input_path)
    LOGGER.info('Working directory: %s', ARGS.work_dir)
    LOGGER.info('Root path: %s', ARGS.root_path)
    LOGGER.info('New spacing: %s', ARGS.spacing)
    LOGGER.info('Weight files: \n%s', '\n'.join([x for x in sorted(WEIGHTS)]))
    LOGGER.info('Cluster correction: %s', ARGS.cluster_correction)
    if ARGS.cluster_correction:
        LOGGER.info('Minimum extent: %s', ARGS.min_extent)
    LOGGER.info('Evaluation: %s', ARGS.evaluate)

    INFERENCE = IndividualInference(ARGS.input_path, ARGS.work_dir)
    INFERENCE.get_data()
    INFERENCE.preprocessing(new_spacing=ARGS.spacing)
    INFERENCE.create_tensors()
    INFERENCE.run_inference(weights=ARGS.weights)
    INFERENCE.save_inference(min_extent=ARGS.min_extent,
                             cluster_correction=ARGS.cluster_correction)

print('Done!')
