import time
import glob
import os
from lung_segmentation.utils import untar, get_weights, create_log
import argparse
from lung_segmentation.run_segmentation import run_segmentation


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str,
                        help=('Existing directory with on CT image to segment per folder.'))
    parser.add_argument('work_dir', type=str,
                        help=('Directory where to store the results.'))

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
            tar_file = get_weights(url, parent_dir)
            untar(tar_file)       
        except:
            logger.error('Pre-trained weights cannot be downloaded. Please check '
                         'your connection and retry or download them manually '
                         'from the repository.')
    else:
        logger.info('Pre-trained network weights found in {}'.format(weights_dir))
    
    weights = [w for w in sorted(glob.glob(os.path.join(parent_dir, 'weights/*.h5')))]
    if len(weights) == 5:
        logger.info('{0} weights files found in {1}. Five fold inference will be calculated.'
                    .format(len(weights), weights_dir))
    elif len(weights) < 5:
        logger.warning('Only {0} weights files found in {1}. There should be 5. Please check '
                       'the repository and download them again in order to run the five fold '
                       'inference will be calculated. The segmentation will still be calculated '
                       'using {0}-fold cross validation but the results might be sub-optimal.'
                       .format(len(weights), weights_dir))
    else:
        logger.error('{} weights file found in {1}. This is not possible since the model was '
                     'trained using a 5-folds cross validation approach. Please check the '
                     'repository and remove all the unknown weights files.'
                     .format(len(weights), weights_dir))
    
    if not os.path.isdir(bin_dir):
        logger.info('No directory containing the binary executable found. '
                    'I will try to download it from the repository.')
        try:
            url = ''
            tar_file = get_weights(url, parent_dir)
            untar(tar_file)
        except:
            logger.error('Binary files cannot be downloaded. Please check '
                         'your connection and retry or download them manually '
                         'from the repository.')
    else:
        logger.info('Binary executables found in {}'.format(bin_dir))
    
    args = parser.parse_args()
    
#     input_dir = '/mnt/sdb/test_lung_seg/'
#     work_dir = '/mnt/sdb/test_lung_seg_out/'

    run_segmentation(args.input_dir, args.work_dir, weights)
#     run_segmentation(input_dir, work_dir, weights)
    
    stop = time.perf_counter()
    logger.info('Process ended after {} seconds!'.format(int(stop-start)))
