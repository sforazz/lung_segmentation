"Simple script to run segmentation inference"
import os
from lung_segmentation.inference import LungSegmentationInference
from lung_segmentation.utils import create_log


INPUT_DIR = '/home/fsforazz/Desktop/PhD_project/fibrosis_project/training_data_MA.xlsx'
WORK_DIR = '/mnt/sdb/tl_mouse_MA_all/'
PARENT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))
LOG_DIR = os.path.join(WORK_DIR, 'logs')

os.environ['bin_path'] = os.path.join(PARENT_DIR, 'bin/')

if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
LOGGER = create_log(LOG_DIR)

inference = LungSegmentationInference(INPUT_DIR, WORK_DIR, deep_check=False)
inference.get_data()
inference.preprocessing(new_spacing=(0.2, 0.2, 0.2))
inference.create_tensors()
inference.run_inference(weights=['/mnt/sdb/tl_mouse_MA/training/double_feat_per_layer_BCE_fold_0.h5'])
inference.save_inference()
inference.run_evaluation()

print('Done!')
