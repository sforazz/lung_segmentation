import requests
import tarfile
import os
from datetime import datetime
import logging
import shutil
import glob
import pydicom
from basecore.utils.dicom import DicomInfo
import pickle
import nrrd
import numpy as np
import cv2
from basecore.utils.filemanip import split_filename
import nibabel as nib
from skimage.filters.thresholding import threshold_otsu
from skimage.transform import resize
import pandas as pd


ALLOWED_EXT = ['.xlsx', '.csv']
ILLEGAL_CHARACTERS = ['/', '(', ')', '[', ']', '{', '}', ' ', '-']


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
    
    return os.path.join(location, 'weights.tar.gz')

 
def untar(fname):

    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall()
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
        dicoms = sorted(glob.glob(raw_data+'/*.dcm'))
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
                folder_name = temp_dir+'/{0}_date_{1}_time_{2}'.format(basename, tag['AcquisitionDate'][0],
                                                                       tag['SeriesTime'][0])
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
        filename = sorted(glob.glob(folder_name+'/*{}'.format(ext)))[0]

    else:
        print('No suitable CT data with name containing "H50s" were found in {}'.format(raw_data))
        filename = None
        folder_name = None

    return filename, folder_name, dcm_info


def save_results(im2save, ref, save_dir=None):

    basedir, basename, ext = split_filename(ref)
    out_basename = basename.split('.')[0]+'_lung_seg'+ext
    if save_dir is not None:
        outname = os.path.join(save_dir, out_basename)
    else:
        outname = os.path.join(basedir, out_basename)
    
    if ext == '.nii' or ext == '.nii.gz':
        ref = nib.load(ref)
        im2save = nib.Nifti1Image(im2save, ref.affine)
        nib.save(im2save, outname)
    elif ext == '.nrrd':
        _, ref_hd = nrrd.read(ref)
        nrrd.write(outname, im2save, header=ref_hd)
    else:
        raise Exception('Extension "{}" is not recognize!'.format(ext))

    return outname


def postprocessing(im2save, method='mouse_fibrosis'):

    if method == 'mouse_fibrosis':
        image = im2save[:, 10:, 10:]
        image = image.reshape(-1, 86, 86)
    elif method=='human':
        image_old = im2save[:, 10:, 10:]
        image_old = image_old.reshape(-1, 86, 86)
        image_old = image_old.reshape(-1, 86, 86)
        image = np.zeros((image_old.shape[0], 512, 512))
        for z in range(image.shape[0]):
            image[z, :, :] = cv2.resize(image_old[z, :, :], (512, 512), interpolation=cv2.INTER_AREA)
        
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)

    return image


def preprocessing(image, label=False, method='mouse_fibrosis'):
    
    image = image.astype('float64')
    if method == 'mouse_fibrosis' or method == 'micro_ct':
        image = cv2.resize(image[:, :], (86, 86),interpolation=cv2.INTER_AREA)
        image = image.reshape(86, 86, 1)
        if not label:
            image -= np.min(image)
            image /= (np.max(image)-np.min(image))
        temp = np.zeros([96, 96, 1])
        temp[10:,10:,:] = image
        image = temp
    elif method == 'human':
        image = cv2.resize(image[:, :], (86, 86),interpolation=cv2.INTER_AREA)
        image = image.reshape(86, 86, 1)
        if not label:
            image -= np.min(image)
            image /= (np.max(image)-np.min(image))
        temp = np.zeros([96, 96, 1])
        temp[10:,10:,:] = image
        image = temp
    
    return image


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
        res = (image - m)/(M - m)
    else:
        raise NotImplementedError('Normalization method called "{}" has not been implemented yet!'.format(method))

    return res


def resize_image(image, order=0, new_spacing=(0.5, 0.5, 0.5)):

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
    new_shape = (image.shape[0]//resampling_factor[0], image.shape[1]//resampling_factor[1], image.shape[2]//resampling_factor[2])
    new_image = resize(image.astype(np.float64), new_shape, order=order, mode='edge', cval=0, anti_aliasing=False)
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
            final_image = resize(final_image, (original_dim[0], original_dim[1]), order=3, mode='edge', cval=0,
                                 anti_aliasing=False)
        if binarize:
            final_image = binarization(final_image)

        return final_image
