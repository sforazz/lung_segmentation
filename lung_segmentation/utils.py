import requests
import tarfile
import os
import sys
from datetime import datetime
import logging
import shutil
import glob
from operator import itemgetter
import collections
import pydicom
import nrrd
import numpy as np
import nibabel as nib
import PySimpleGUI as sg
from skimage.filters.thresholding import threshold_otsu
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.cbook as cbook
import subprocess as sp
from medpy.metric.binary import hd, hd95, dc
from lung_segmentation.configuration import (
        STANDARD_CONFIG, HIGH_RES_CONFIG, HUMAN_CONFIG, CUSTOM_CONFIG)


ALLOWED_EXT = ['.xlsx', '.csv']
ILLEGAL_CHARACTERS = ['/', '(', ')', '[', ']', '{', '}', ' ', '-']


class DicomInfo(object):
    
    def __init__(self, dicoms):

        if type(dicoms) == list:
            self.dcms = dicoms
        elif os.path.isdir(dicoms):
            dcms = list(dicoms.glob('*.dcm'))
            if not dcms:
                dcms = list(dicoms.glob('*.IMA'))
            if not dcms:
                raise Exception('No DICOM files found in {}'.format(dicoms))
            else:
                self.dcms = dcms
        else:
            self.dcms = [dicoms]

    def get_tag(self, tag):
        
        tags = {}

        if type(tag) is not list:
            tag = [tag]
        for t in tag:
            values = []
            for dcm in self.dcms:
                header = pydicom.read_file(str(dcm))
                try:
                    val = header.data_element(t).value
                    if isinstance(val, collections.Iterable) and type(val) is not str:
                        val = tuple(val)
                    else:
                        val = str(val)
                    values.append(val)
                except (AttributeError, KeyError):
                    print ('{} seems to do not have the requested DICOM field ({})'.format(dcm, t))

            tags[t] = list(set(values))
        
        return self.dcms, tags

    def check_uniqueness(self, InstanceNums, SeriesNums):
        
        toRemove = []
        if (len(InstanceNums) == 2*(len(set(InstanceNums)))) and len(set(SeriesNums)) == 1:
            sortedInstanceNums = sorted(zip(self.dcms, InstanceNums), key=itemgetter(1))
            uniqueInstanceNums = [x[0] for x in sortedInstanceNums[:][0:-1:2]]
            toRemove = toRemove+uniqueInstanceNums
        
        return toRemove


def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.
    Parameters
    ----------
    fname : str
        file or path name
    Returns
    -------
    pth : str
        base path from fname
    fname : str
        filename from fname, without extension
    ext : str
        file extension from fname
    """

    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]

    pth = os.path.dirname(fname)
    fname = os.path.basename(fname)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and \
                (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = os.path.splitext(fname)

    return pth, fname, ext


def get_files(url, location, file, ext='.tar.gz'):

    r = requests.get(url)
    with open(os.path.join(location, file+ext), 'wb') as f:
        f.write(r.content)
    print(r.status_code)
    print(r.headers['content-type'])
    print(r.encoding)

    return os.path.join(location, file+ext)


def untar(fname):

    untar_dir = os.path.split(fname)[0]
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname)
        tar.extractall(path=untar_dir)
        tar.close()
        print("Extracted in Current Directory")
    else:
        print("Not a tar.gz file: {}".format(fname))


def create_log(log_dir):

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logger = logging.getLogger('lungs_segmentation')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(
        log_dir, 'lungs_segmentation_{}.log'.format(dt_string)))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def batch_processing(input_data, key_col1='subjects', key_col2='masks', root=''):
    """Function to process the data in batch mode. It will take a .csv or .xlsx file with
    two columns. The first one called 'subjects' contains all the paths to the raw_data folders
    (one path per raw); the second one called 'masks' contains all the corresponding paths to 
    the segmented mask folders. 
    Parameters
    ----------
    input_data : str
        Excel or CSV file
    root : str
        (optional) root path to pre-pend to each subject and mask in the input_data file
    Returns
    -------
    raw_data : list
        list with all the subjects to process
    masks : list
        list with the corresponding mask to use to extract the features
    """
    if os.path.isfile(input_data):
        _, _, ext = split_filename(input_data)
        if ext not in ALLOWED_EXT:
            raise Exception('The file extension of the specified input file ({}) is not supported.'
                            ' The allowed extensions are: .xlsx or .csv')
        if ext == '.xlsx':
            files = pd.read_excel(input_data)
        elif ext == '.csv':
            files = pd.read_csv(input_data)
        files=files.dropna()
        try:
            masks = [os.path.join(root, str(x)) for x in list(files[key_col2])]
        except KeyError:
            print('No "masks" column found in the excel sheet. The cropping, if selected, will be performed without it.')
            masks = None
        raw_data = [os.path.join(root, str(x)) for x in list(files[key_col1])] 

        return raw_data, masks


def dicom_check(raw_data, temp_dir, deep_check=True):
    """Function to arrange the mouse lung data into a proper structure.
    In particular, this function will look into each raw_data folder searching for
    the data with H50s in the series description field in the DICOM header. Then,
    it will copy those data into another folder and will return the path to the first
    DICOM file that will be used to run the DICOM to NRRD conversion.
    Parameters
    ----------
    raw_data : str
        path to the raw data folder 
    Returns
    -------
    pth : str
        path to the first DICOM volume
    """
    
    dicoms = sorted(glob.glob(raw_data+'/*.IMA'))
    dcm_info = {}
    processed = False
    if not dicoms:
        dicoms = sorted(glob.glob(raw_data+'/*'))
        if not dicoms:
            raise Exception('No DICOM files found in {}! Please check.'.format(raw_data))
        else:
            ext = '.dcm'
    else:
        ext = '.IMA'

    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    basename = raw_data.split('/')[-1]
    for character in ILLEGAL_CHARACTERS:
        basename = basename.replace(character, '_')

    sequence_numbers = list(set([str(pydicom.read_file(x).SeriesNumber) for x in dicoms]))
    if len(sequence_numbers) > 1 and deep_check:
        for n_seq in sequence_numbers:
            dicom_vols = [x for x in dicoms if n_seq==str(pydicom.read_file(x).SeriesNumber)]
            dcm_hd = pydicom.read_file(dicom_vols[0])
            if len(dicom_vols) > 1 and '50s' in dcm_hd.SeriesDescription and not processed:
                dcm = DicomInfo(dicom_vols)
                _, tag = dcm.get_tag(['AcquisitionDate', 'SeriesTime'])
                folder_name = temp_dir+'/{0}_date_{1}_time_{2}'.format(
                    basename, tag['AcquisitionDate'][0], tag['SeriesTime'][0])
                slices = [pydicom.read_file(x).InstanceNumber for x in dicom_vols]
                if len(slices) != len(set(slices)):
                    print('Duplicate slices found in {} for H50s sequence. Please check. '
                          'This subject will be excluded from the analysis.'.format(raw_data))
                    continue
                processed = True
                break
    elif len(sequence_numbers) == 1:
        folder_name = temp_dir+'/{0}'.format(basename)
        dicom_vols = dicoms
        processed = True

    if processed:
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        else:
            shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        for x in dicom_vols:
            try:
                shutil.copy2(x, folder_name)
            except:
                continue
        filename = sorted(glob.glob(folder_name+'/*'.format(ext)))[0]

    else:
        print('No suitable CT data with name containing "H50s" were found in {}'.format(raw_data))
        filename = None
        folder_name = None

    return filename, folder_name, dcm_info


def binarization(image):

    th = threshold_otsu(image)
    image[image>=th] = 1
    image[image!=1] = 0

    return image


def normalize(image, method='zscore'):

    image = np.asanyarray(image)
    image = image.astype('float64')
    if method == 'zscore':
        mns = image.mean()
        sstd = image.std()
        res = (image - mns)/sstd
    elif method == '0-1':
        m = image.min()
        M = image.max()
        if (M - m) > 0:
            res = (image - m)/(M - m)
        else:
            res = image
    else:
        raise NotImplementedError('Normalization method called "{}" has not been implemented yet!'.format(method))

    return res


def resize_image(image, order=0, new_spacing=(0.1, 0.1, 0.1), save2file=True):

    basepath, fname, ext = split_filename(image)
    outname = os.path.join(basepath, fname+'_resampled'+ext)
    if ext == '.nrrd':
        image, hd = nrrd.read(image)
        space_x = np.abs(hd['space directions'][0, 0])
        space_y = np.abs(hd['space directions'][1, 1])
        space_z = np.abs(hd['space directions'][2, 2])
    elif ext == '.nii.gz' or ext == '.nii':
        hd = nib.load(image).header
        affine = nib.load(image).affine
        image = nib.load(image).get_data()
        space_x, space_y, space_z = hd.get_zooms()

    resampling_factor = (new_spacing[0]/space_x, new_spacing[1]/space_y, new_spacing[2]/space_z)
    new_shape = (image.shape[0]//resampling_factor[0], image.shape[1]//resampling_factor[1],
                 image.shape[2]//resampling_factor[2])
    new_image = resize(image.astype(np.float64), new_shape, order=order, mode='edge',
                       cval=0, anti_aliasing=False)
    if save2file:
        if ext == '.nrrd':
            hd['sizes'] = np.array(new_image.shape)
            hd['space directions'][0, 0] = new_spacing[0]
            hd['space directions'][1, 1] = new_spacing[1]
            hd['space directions'][2, 2] = new_spacing[2]
            nrrd.write(outname, new_image, header=hd)
        elif ext == '.nii.gz' or ext == '.nii':
            im2save = nib.Nifti1Image(new_image, affine=affine)
            nib.save(im2save, outname)

        return new_image, tuple(map(int, new_shape)), outname, image.shape
    else:
        return new_image, np.mean(resampling_factor)


def save_prediction_2D(generated_images, dict_val, binarize=False):

    n_images = len(dict_val)
    batches = generated_images.shape[0]//n_images
    for n in range(n_images):
        im_shape = dict_val[n]['im_size']
        indexes = dict_val[n]['indexes']
        image = generated_images[n*batches:(n+1)*batches, :]
        final_image = np.zeros((im_shape[0], im_shape[1], batches), dtype=np.float16)-2
        k = 0
        for j in indexes[1]:
            for i in indexes[0]:
                final_image[i[0]:i[1], j[0]:j[1], k] = image[k, :, :, 0]
                k += 1
        final_image[final_image==-2] = np.nan
        final_image = np.nanmean(final_image, axis=2)
        final_image[np.isnan(final_image)] = 0
        final_image = final_image.astype(np.float32)
        original_dim = dict_val[n]['orig_dim']
        if tuple(final_image.shape) != original_dim:
            final_image = resize(final_image, (original_dim[0], original_dim[1]),
                                 order=3, mode='edge', cval=0,
                                 anti_aliasing=False)
        if binarize:
            final_image = binarization(final_image)

        return final_image


def dice_calculation(gt, seg):

    gt, hd_gt = nrrd.read(gt)
    seg, hd_seg = nrrd.read(seg)
    if (hd_seg['space origin'] == hd_gt['space origin']).all():
        seg = np.squeeze(seg)
        gt = gt.astype('uint16')
        seg = seg.astype('uint16')
        dice = dc(seg, gt)
    else:
        dice = None
#     vox_gt = np.sum(gt) 
#     vox_seg = np.sum(seg)
#     try:
#         common = np.sum(gt & seg)
#     except:
#         print(gt)
#     dice = (2*common)/(vox_gt+vox_seg) 
    return dice


def violin_box_plot(to_plot, outname):

    fig = plot.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    parts=ax.violinplot(to_plot, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('xkcd:lightblue')
        pc.set_edgecolor('xkcd:blue')
        pc.set_alpha(1)
        pc.set_linewidth(2)

    medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
    stats = cbook.boxplot_stats(to_plot)
    flierprops = dict(marker='o', markerfacecolor='green', markersize=12, linestyle='none')
    ax.set_axisbelow(True)
    plot.gca().yaxis.grid(True, ls='-', color='white')
    ax.bxp(stats, flierprops=flierprops, medianprops=medianprops)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.set_facecolor('lightgrey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plot.savefig(outname)
    plot.close()


def run_cluster_correction(image, th=0.5, min_extent=10000):

    out_nii = image.split('.nrrd')[0]+'.nii.gz'
    out_image = out_nii.split('.nii.gz')[0]+'_cc.nii.gz'
    out_text = out_nii.split('.nii.gz')[0]+'_cc.txt'
    outname = out_nii.split('.nii.gz')[0]+'_corrected.nii.gz'
    outname_nrrd = out_nii.split('.nii.gz')[0]+'_corrected.nrrd'

    im_nrrd, header_nrrd = nrrd.read(image)
    nii2save = nib.Nifti1Image(im_nrrd, affine=np.eye(4))
    nib.save(nii2save, out_nii)

    cmd = 'cluster -i {0} -t {1} -o {2} --minextent={3} --olmax={4}'.format(
        out_nii, th, out_image, min_extent, out_text)
    _ = sp.check_output(cmd, shell=True)
    with open(out_text, 'r') as f:
        res = [x.split() for x in f]
    mat = np.asarray(res[1:]).astype('float')
    try:
        clusters = list(set(mat[mat[:, 1] > 0.95][:, 0]))
    except:
        print()
    add_cmd = None
    for i, cl in enumerate(clusters):
        if len(clusters) > 2:
            print('Found {0} clusters for image {1}, please check because the'
                  ' usual number of clusters should be not greater than 2.'.format(len(clusters), image))
        out_cl = image.split('.nii.gz')[0]+'_cc_{}.nii.gz'.format(cl)
        if len(clusters) > 1:
            cmd = 'fslmaths {0} -uthr {1} -thr {1} -bin {2}'.format(out_image, cl, out_cl)
        else:
            cmd = 'fslmaths {0} -uthr {1} -thr {1} -bin {2}'.format(out_image, cl, outname)
        sp.check_output(cmd, shell=True)
        if len(clusters) > 1:
            if i == 0:
                add_cmd = 'fslmaths {} -add'.format(out_cl)
            elif i == len(clusters)-1:
                add_cmd = add_cmd + ' {0} {1}'.format(out_cl, outname)
            else:
                add_cmd = add_cmd + ' {0} -add'.format(out_cl)
    if add_cmd is not None:
        sp.check_output(add_cmd, shell=True)

    nii_im = nib.load(outname).get_data()
    nrrd.write(outname_nrrd, nii_im, header=header_nrrd)

    return outname_nrrd


def run_hd(image1, image2, mode='max'):

    image1_data, hd1 = nrrd.read(image1)
    space_x = np.abs(hd1['space directions'][0, 0])
    space_y = np.abs(hd1['space directions'][1, 1])
    space_z = np.abs(hd1['space directions'][2, 2])
    image2_data, hd2 = nrrd.read(image2)
    if (hd1['space origin'] == hd2['space origin']).all():
        if mode == 'max':
            hd_val = hd(image1_data, image2_data, (space_x, space_y, space_z))
        elif mode == '95':
            hd_val = hd95(image1_data, image2_data, (space_x, space_y, space_z))
        else:
            raise Exception('Unknown mode "{}". Possible values are "max" or "95".'
                            .format(mode))
    else:
        hd_val = None
    return hd_val


def build_gui():
    "Function to build the GUI for inference"
    configured = False

    while not configured:
        disable_dw_weights = False
        sg.ChangeLookAndFeel('GreenTan')

        layout = [
            [sg.Text('Lung Segmentation using CNN', size=(30, 1), font=("Helvetica", 25))],
            [sg.Text('Please choose a configuration file based on your data:')],
            [sg.Listbox(values=('Low resolution mouse', 'High resolution mouse', 'Human',
                                'Custom configuration'),
                        size=(30, 3), default_values='Low resolution mouse')],
            [sg.Submit(), sg.Quit()]
        ]
        window = sg.Window('SIENA', default_element_size=(40, 1)).Layout(layout)
        config_button, values = window.Read()
        if config_button == 'Quit':
            sys.exit()
        else:
            window.close()
        if values[0][0] == 'Low resolution mouse':
            config = STANDARD_CONFIG
            app_name = 'Lung segmentation inference for low resolution mouse CT images'
        elif values[0][0] == 'High resolution mouse':
            config = HIGH_RES_CONFIG
            app_name = 'Lung segmentation inference for high resolution mouse CT images'
        elif values[0][0] == 'Human':
            config = HUMAN_CONFIG
            app_name = 'Lung segmentation inference for human CT images'
        else:
            config = CUSTOM_CONFIG
            disable_dw_weights = True
            app_name = 'Lung segmentation inference using custom settings'

        dc_check = config['dicom_check']
        new_spacing = config['spacing']
        min_extent = config['min_extent']
        cluster_correction = config['cluster_correction']
        weights_url = config['weights_url']

        input_layout = [
            [sg.Text('Input path', size=(15, 1), auto_size_text=False, justification='right'),
             sg.InputText('', key='in_path', tooltip=(
                 'Input can be either a folder (with one or more sub-folder(s) \n'
                 'for each CT image) or an Excel sheet with one image path \n'
                 'per raw. See the documentation for more information.')),
             sg.FolderBrowse()],
            [sg.Text('Root path', size=(15, 1), auto_size_text=False, justification='right'),
             sg.InputText('', key='root_path', tooltip=(
                 'If an Excel sheet was specified as input, then the root path \n'
                 'is the path that will be prepended to each raw in the \n'
                 'input file.')),
             sg.FolderBrowse()],
            [sg.Text('Working directory', size=(15, 1), auto_size_text=False,
                     justification='right'),
             sg.InputText('', key='work_dir', tooltip=(
                 'Path where to save the results. If the directory does not \n'
                 'exist, it will be created.')),
             sg.FolderBrowse()],
            [sg.Checkbox('Download weights', change_submits = True, enable_events=True,
                         default=True, key='download_weights', disabled=disable_dw_weights,
                         tooltip='Whether or not to download the weights from the \n'
                         'repository. It does not work for custom settings.')],
            [sg.Text('Weights directory', size=(15, 1), auto_size_text=False,
                     justification='right'),
             sg.InputText('', disabled = not disable_dw_weights, key='weights_dir',
                          tooltip=('Specify here the path to the folder containing the \n'
                                   'network weights. If you do not have them, please tick \n'
                                   'the "Download weights" box above and the application \n'
                                   'will try to download them from the repository.\n'
                                   'N.B. The download does not work for custom settings.')),
             sg.FolderBrowse(disabled = not disable_dw_weights, key='weights_dir_2')]
            ]

        colx = [[sg.Text('Spacing X')],
                [sg.Slider(range=(0.0, 5.0), orientation='v', resolution=0.05, size=(5, 20),
                           default_value=new_spacing[0],
                           tooltip='New resolution (in mm) along x direction.',
                           key='space_x')]]
        coly = [[sg.Text('Spacing Y')],
                [sg.Slider(range=(0.0, 5.0), orientation='v', resolution=0.05, size=(5, 20),
                           default_value=new_spacing[1],
                           tooltip='New resolution (in mm) along y direction.',
                           key='space_y')]]
        colz = [[sg.Text('Spacing Z')],
                [sg.Slider(range=(0.0, 5.0), orientation='v', resolution=0.05, size=(5, 20),
                           default_value=new_spacing[2],
                           tooltip='New resolution (in mm) along z direction.',
                           key='space_z')]]
        preproc_layout = [
            [sg.Checkbox('DICOM check', default=dc_check,
                         tooltip='Whether or not to carefully check the DICOM header. \n'
                                 'This check is based on our low resolution mouse data \n'
                                 'and might be too stringent for data coming from \n'
                                 'different sites and/or acquired with different resolution.\n'
                                 'If the last is the case, then turn this check off.',
                         key='dcm_check')],
            [sg.Text('New spacing (for resampling)',
                     tooltip='Here you can specify the voxel size (in mm) \n'
                             'that will be used to resample the image before\n'
                             'running the inference. The network has been trained\n'
                             'using images with a defined resolution. The default\n'
                             'values for the resolution have been stored in the 3 \n'
                             'pre-defined configuration files, however, \n'
                             'if the network was re-trained with different\n'
                             'resolution, you can simply change it here.')],
            [sg.Column(colx), sg.Column(coly), sg.Column(colz)]]

        post_proc_layout = [
            [sg.Radio('Cluster correction', 'pp', default=cluster_correction,
                      key='cluster_correction', change_submits = True, enable_events=True,
                      tooltip=('This correction should be more accurate wrt the standard\n'
                               'binarization and it is recommended for high res and human \n'
                               'segmentation because it should remove all the small isolated \n'
                               'wrongly segmented areas. However is more time consuming and \n'
                               'relies on external software (FSL), so run it only if needed.')),
             sg.Text('Minimum extent', size=(15, 1), auto_size_text=False, justification='right',
                     tooltip=('If "Cluster correction" is selected, then here you can \n'
                              'specify the minimum number of voxels that a cluster can \n'
                              'have in order to be considered as possible lung. This value \n'
                              'depends on the resolution and size of the CT image. Usually \n'
                              'for low res mice 350 is enough, while for high res mice and \n'
                              'humans can be set to 100000 and 300000, respectively.')),
             sg.InputText(min_extent, key='min_extent', disabled = not cluster_correction,
                          size=(15, 10))],
            [sg.Radio('Otsu binarizazion', 'pp', default=not cluster_correction,
                      change_submits = True, enable_events=True,
                      tooltip=('If selected, the final segmented image will be binarized\n'
                               'using the Otsu thresholding method.'))],
            [sg.Checkbox('Run segmentation evaluation', default=False,
                         tooltip=('If ground truth segmentations are available, \n'
                                  'you can specify them in the Excel sheet and select\n'
                                  'this in order to evaluate the CNN segmentation. Both\n'
                                  'Dice score and Hausdorff distance will be calculated.'),
                         key='evaluate')]]

        layout = [
            [sg.Text(app_name, size=(55, 1), font=("Helvetica", 20))],
            [sg.Frame('Input and output specs', input_layout, font='Any 12', title_color='black')],
            [sg.Text('_'  * 80)],
            [sg.Frame('Pre-processing specs', preproc_layout, font='Any 12', title_color='black')],
            [sg.Text('_'  * 80)],
            [sg.Frame('Post-processing specs', post_proc_layout, font='Any 12',
                      title_color='black')],
            [sg.Submit(), sg.Quit(), sg.Cancel()]]
        main_window = sg.Window('SIENA2', default_element_size=(50, 1)).Layout(layout)
        while True:
            main_button, values = main_window.Read()
            if weights_url is not None:
                if values['download_weights'] and weights_url is not None:
                    main_window['weights_dir'].Update(disabled = True)
                    main_window['weights_dir_2'].Update(disabled = True)
                if not values['download_weights'] and weights_url is not None:
                    main_window['weights_dir'].Update(disabled = False)
                    main_window['weights_dir_2'].Update(disabled = False)
            if values['cluster_correction']:
                main_window['min_extent'].Update(disabled = False)
            if not values['cluster_correction']:
                main_window['min_extent'].Update(disabled = True)
            if main_button == 'Quit':
                sys.exit()
            elif main_button == 'Cancel':
                main_window.close()
                break
            elif main_button == 'Submit':
                configured = True
                break
    values['weights_url'] = weights_url
    return values
