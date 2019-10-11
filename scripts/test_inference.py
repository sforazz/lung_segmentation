"Simple script to run segmentation inference"
import os
import argparse
from lung_segmentation.utils import create_log
from lung_segmentation.inference import LungSegmentationInference


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--input_path', '-i', type=str,
                        help=('Existing Excel file with a list of all the folders containing DICOMS'
                              ' that will be used for training the network.'))
    PARSER.add_argument('--root_path', '-r', type=str, default='',
                        help=('Path that has to be appended to all the folders in the input_file.'))
    PARSER.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--min-extent', type=int, default=400,
                        help=('Minimum lung extension (in voxels) that will be used to '
                              'run the final correction after the inference. For mouse acquired '
                              'with clinical CT, 400 should be enough, while for micro-CT or human '
                              'date this should be set to 30000-40000. Default is 400.'))
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

    ARGS = PARSER.parse_args()

    PARENT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))
    os.environ['bin_path'] = os.path.join(PARENT_DIR, 'bin/')

    LOG_DIR = os.path.join(PARENT_DIR, 'logs')
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    LOGGER = create_log(LOG_DIR)

    if ARGS.spacing is not None:
        NEW_SPACING = (ARGS.spacing[0], ARGS.spacing[1], ARGS.spacing[2])
    else:
        NEW_SPACING = None

    INFERENCE = LungSegmentationInference(ARGS.input_path, ARGS.work_dir, deep_check=ARGS.dcm_check)
    INFERENCE.get_data(root_path=ARGS.root_path)
    INFERENCE.preprocessing(new_spacing=NEW_SPACING)
    INFERENCE.create_tensors()
    INFERENCE.run_inference(weights=ARGS.weights)
    INFERENCE.save_inference(min_extent=ARGS.min_extent)
    INFERENCE.run_evaluation(new_spacing=(3, 3, 3))

print('Done!')
