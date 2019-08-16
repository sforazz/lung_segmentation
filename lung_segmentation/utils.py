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


ALLOWED_EXT = ['.xlsx', '.csv']
ILLEGAL_CHARACTERS = ['/', '(', ')', '[', ']', '{', '}', ' ', '-']


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

    
def dicom_check(raw_data, temp_dir, deep_check=True):
    """Function to arrange the mouse lung data into a proper struture.
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
    for n_seq in sequence_numbers:                     
        dicom_vols = [x for x in dicoms if n_seq==str(pydicom.read_file(x).SeriesNumber)]
        if not deep_check:
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
        else:
            folder_name = temp_dir+'/{0}'.format(basename)
            
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
        if not deep_check:
            with open(folder_name+'/dcm_info.p', 'wb') as fp:
                pickle.dump(tag, fp, protocol=pickle.HIGHEST_PROTOCOL)
        processed = True
    if not processed:
        print('No suitable CT data with name containing "H50s" were found in {}'.format(raw_data))
        filename = None

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
