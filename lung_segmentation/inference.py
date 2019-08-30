from lung_segmentation.base import LungSegmentationBase
from pathlib import Path
import logging
import numpy as np
from lung_segmentation.models import unet_lung
import os
import glob


logger = logging.getLogger('lungs_segmentation')


class LungSegmentationInference(LungSegmentationBase):
    
    def get_data(self):

        if (os.path.isdir(os.path.join(self.work_dir, 'testing'))
                and os.path.isfile(os.path.join(self.work_dir, 'testing', 'test_subjects.txt'))):
            with open(os.path.join(self.work_dir, 'testing', 'test_subjects.txt'), 'r') as f:
                self.dcm_folders = [x.strip() for x in f]
        else:
            logger.info('No folder called "testing" in the working directory.'
                        ' The pipeline will look for DICOM file to use for '
                        'inference in all the sub-folders within the '
                        'working directory.')
            input_dir = Path(self.input_dir)
            logger.info('Input dir: {}'.format(input_dir))
            self.dcm_folders = sorted([input_dir/x for x in input_dir.iterdir() if x.is_dir() and 
                                      ((input_dir/x).glob('*.dcm') or (input_dir/x).glob('*.DCM')
                                       or (input_dir/x).glob('*.IMA'))])
            logger.info('Found {0} sub-folders in {1}. They will be used to run the inference.'
                        .format(len(self.dcm_folders), str(input_dir)))

        self.work_dir = os.path.join(self.work_dir, 'testing')
    
    def create_tensors(self, patch_size=(96, 96), save2npy=False):
        return LungSegmentationBase.create_tensors(self, patch_size=patch_size, save2npy=save2npy)
    
    def run_inference(self, weights):
        
        test_set = np.asarray(self.image_tensor)
        predictions = []
        logger.info('Segmentation inference started.')
        model = unet_lung()
        for i, w in enumerate(weights):
            logger.info('Segmentation inference fold {}.'.format(i+1))
            model.load_weights(w)
            predictions.append(model.predict(test_set))
            
        predictions = np.asarray(predictions, dtype=np.float32)
        prediction = np.mean(predictions, axis=0)
    