from lung_segmentation.training import LungSegmentationTraining
import os
from lung_segmentation.utils import create_log

# cropping = ImageCropping('/mnt/sdb/test_lung_seg_out/CT_1_date_20160323_time_171726.234000/TR6_20W_RT14_(RT+FG+N)_B_MAHMOUD.nrrd')
# to_segment = cropping.crop_wo_mask()
# t = load_data_2D('data_dir', 'data_type', data_list=['/mnt/sdb/test_lung_seg_out/Spiegelberg_A1_2015.10.02/slice_0000_cropped_mouse_0.nrrd'], img_size=(170, 170), patch_size=(96, 96))
input_dir = '/home/fsforazz/Desktop/PhD_project/fibrosis_project/mouse_list_endo.xlsx'
work_dir = '/mnt/sdb/endo_no_mask/'
parent_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))

os.environ['bin_path'] = os.path.join(parent_dir, 'bin/')
log_dir = os.path.join(work_dir, 'logs')
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

logger = create_log(log_dir)

ls = LungSegmentationTraining(input_dir, work_dir, deep_check=False, tl=False)
ls.get_data(root_path='/mnt/sdb/mouse_fibrosis_data/', testing=False)
ls.preprocessing(new_spacing=(1, 1, 1))
# ls.create_tensors()
# ls.data_split(additional_dataset=['/mnt/sdb/tl_mouse_MA_all/training', '/mnt/sdb/human_seg/training'])
# ls.run_training(n_epochs=100)
