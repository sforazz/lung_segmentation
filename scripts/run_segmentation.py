import time
import glob
import os
from lung_segmentation.utils import untar, get_files, create_log
import argparse
from lung_segmentation.pipeline import pipeline


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', '-i', type=str,
                        help=('Existing directory with on CT image to segment per folder.'))
    parser.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
    parser.add_argument('--no-dcm-check', action='store_true',
                        help=('Whether or not to carefully check the DICOM header. '
                              'This check is based on our data and might too stringent for other'
                              ' dataset, so you might want to turn it off. If you turn it off, '
                              'please make sure that the DICOM data are correct. The check is '
                              'performed by default.'))
    
    args = parser.parse_args()

    parent_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))
    os.environ['bin_path'] = os.path.join(parent_dir, 'bin/')
    
    log_dir = os.path.join(parent_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    
    logger = create_log(log_dir)

    start = time.perf_counter()
    logger.info('Process started!')

    weights_dir = os.path.join(parent_dir, 'weights')
    bin_dir = os.path.join(parent_dir, 'bin')

    if not os.path.isdir(weights_dir):
        logger.info('No pre-trained network weights, I will try to download them.')
        try:
            url = ''
            tar_file = get_files(url, parent_dir, 'weights')
            untar(tar_file)       
        except:
            logger.error('Pre-trained weights cannot be downloaded. Please check '
                         'your connection and retry or download them manually '
                         'from the repository.')
            raise Exception('Unable to download network weights!')
    else:
        logger.info('Pre-trained network weights found in {}'.format(weights_dir))
    
    weights = [w for w in sorted(glob.glob(os.path.join(parent_dir, 'weights/*.h5')))]
    if len(weights) == 5:
        logger.info('{0} weights files found in {1}. Five folds inference will be calculated.'
                    .format(len(weights), weights_dir))
    elif len(weights) < 5:
        logger.warning('Only {0} weights files found in {1}. There should be 5. Please check '
                       'the repository and download them again in order to run the five folds '
                       'inference will be calculated. The segmentation will still be calculated '
                       'using {0}-folds cross validation but the results might be sub-optimal.'
                       .format(len(weights), weights_dir))
    else:
        logger.error('{} weights file found in {1}. This is not possible since the model was '
                     'trained using a 5-folds cross validation approach. Please check the '
                     'repository and remove all the unknown weights files.'
                     .format(len(weights), weights_dir))
    
    if not os.path.isdir(bin_dir):
        logger.info('No directory containing the binary executables found. '
                    'I will try to download it from the repository.')
        try:
            url = ''
            tar_file = get_files(url, parent_dir, 'bin')
            untar(tar_file)
        except:
            logger.error('Binary files cannot be downloaded. Please check '
                         'your connection and retry or download them manually '
                         'from the repository.')
            raise Exception('Unable to download binary files!')
    else:
        logger.info('Binary executables found in {}'.format(bin_dir))


    pipeline(args.input_dir, args.work_dir, weights, deep_check=args.no_dcm_check)
    
    stop = time.perf_counter()
    logger.info('Process successfully ended after {} seconds!'.format(int(stop-start)))
