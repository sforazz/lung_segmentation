import numpy as np
import nibabel as nib
import os
import glob
from lung_segmentation.utils import normalize
import nrrd


def load_data_2D(data_dir, data_type, data_list=[], array=None, mb=[], bs=None, init=None, prediction=False,
                 img_size=(192, 192), patch_size=(96, 96), binarize=False, normalization=True, result_dict=None):
    if array is not None:
        data_list = [1]
    else:
        if data_list:
            data_list = data_list
        elif bs is not None and init is not None:
            data_list = sorted(glob.glob(os.path.join(data_dir, data_type)))[init:bs]
        else:
            data_list = sorted(glob.glob(os.path.join(data_dir, data_type)))

    
    patch_width = patch_size[0]
    patch_height = patch_size[1]

    dx = img_size[0] if img_size[0] >= patch_width else patch_width
    dy = img_size[1] if img_size[1] >= patch_height else patch_height
    
    if len(mb) < 2:
        mb.append(dx//patch_width)

    if len(mb) < 2:
        mb.append(dy//patch_height)
    
    diffX = dx - patch_width if dx - patch_width != 0 else dx
    diffY = dy - patch_height if dy - patch_height != 0 else dy

    overlapX = diffX//(mb[0]-1) if not dx % patch_width and mb[0] > 1 else diffX//(mb[0])
    overlapY = diffY//(mb[1]-1) if not dy % patch_height and mb[1] > 1 else diffY//(mb[1])
    
    indX = 0
    xx = []
    while indX+patch_width <= dx:
        xx.append([indX, indX+patch_width])
        indX = indX + overlapX
    
    indY = 0
    yy = []
    while indY+patch_height <= dy:
        yy.append([indY, indY+patch_height])
        indY = indY + overlapY

    final_array = None

    for index in range(len(data_list)):

        if array is not None:
            array_orig = array
        else:
            data_path = data_list[index]
            array_orig, _ = nrrd.read(data_path)
        if normalization:
            array_orig = normalize(array_orig, method='0-1')
        if binarize:
            array_orig[array_orig != 0] = 1

        original_size = array_orig.shape

        if img_size[0] < patch_width or img_size[1] < patch_height:
            delta_x = (patch_width - img_size[0]) if (img_size[0] < patch_width) else 0
            delta_y = (patch_height - img_size[1]) if img_size[1] < patch_height else 0
            new_x = patch_width if (img_size[0] < patch_width) else img_size[0]
            new_y = patch_height if (img_size[1] < patch_height) else img_size[1]
            if len(array_orig.shape) == 3:
                temp = np.zeros([new_x, new_y, array_orig.shape[2]])
                temp[delta_x:, delta_y:, :] = array_orig
            else:
                try:
                    temp = np.zeros([new_x, new_y])
                    temp[delta_x:, delta_y:] = array_orig
                except:
                    print()
            array_orig = temp
        else:
            delta_x = 0
            delta_y = 0
        
        data_array = [array_orig[i[0]:i[1], j[0]:j[1]] for j in yy for i in xx]
        data_array = np.asarray(data_array, dtype=np.float16)
#         if normalization:
#             data_array = normalize(data_array, method='0-1')
#         if binarize:
#             data_array[data_array != 0] = 1

        arrays = data_array.reshape((-1, patch_width, patch_height, 1))

        if final_array is not None:
                    final_array = np.concatenate([final_array, arrays], axis=0)
        else:
                    final_array = arrays
        if result_dict is None:
            results_dict = {}
        if prediction:
            results_dict[index] = {}
            results_dict[index]['image_dim'] = original_size
            results_dict[index]['indexes'] = [xx, yy]
#             results_dict[index]['im_size'] = [dx, dy]
            results_dict[index]['deltas'] = [delta_x, delta_y]
            results_dict[index]['patches'] = final_array.shape[0]


    return final_array, results_dict
