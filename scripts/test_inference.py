from lung_segmentation.inference import LungSegmentationInference
import os
from lung_segmentation.utils import create_log


input_dir = '/home/fsforazz/Desktop/PhD_project/fibrosis_project/mouse_list_all.xlsx'
work_dir = '/mnt/sdb/test_ls/'
parent_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))

os.environ['bin_path'] = os.path.join(parent_dir, 'bin/')
log_dir = os.path.join(work_dir, 'logs')
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

logger = create_log(log_dir)

ls = LungSegmentationInference(input_dir, work_dir, deep_check=True)
ls.get_data()
ls.preprocessing()
ls.create_tensors()
ls.run_inference(weights='')