from .base import BaseConverter
import subprocess as sp
import os
import glob


class DicomConverter(BaseConverter):
    
    def convert(self, convert_to='nifti_gz', method='dcm2niix', force=False):

        if convert_to == 'nrrd':
            print('\nConversion from DICOM to NRRD...')
            ext = '.nrrd'
            if method=='dcm2niix':
                cmd = ("dcm2niix -o {0} -f {1} -e y {2}".format(self.basedir, self.filename, self.basedir))
            elif method=='slicer':
                cmd = (('Slicer --no-main-window --python-code '+'"node=slicer.util.loadVolume('+
                    "'{0}', returnNode=True)[1]; slicer.util.saveNode(node, '{1}'); exit()"+'"')
                    .format(self.toConvert, os.path.join(self.basedir, self.filename)+ext))

            elif method=='mitk':
                cmd = ("MitkCLDicom2Nrrd -i '{0}' -o '{1}'".format(self.toConvert,
                                                                   os.path.join(self.basedir, self.filename)+ext))

            else:
                raise Exception('Not recognized {} method to convert from DICOM to NRRD.'.format(method))
        elif convert_to == 'nifti_gz':
            print('\nConversion from DICOM to NIFTI_GZ...')
            ext = '.nii.gz'
            if method == 'dcm2niix':
                if force:
                    cmd = ("dcm2niix -o {0} -f {1} -z y -p n -m y {2}".format(self.basedir, self.filename,
                                                                         self.toConvert))
                else:
                    cmd = ("dcm2niix -o {0} -f {1} -z y -p n {2}".format(self.basedir, self.filename,
                                                                         self.toConvert))
            else:
                raise Exception('Not recognized {} method to convert from DICOM to NIFTI_GZ.'.format(method))
        else:
            raise NotImplementedError('The conversion from DICOM to {} has not been implemented yet.'
                                      .format(convert_to))
        try:
            sp.check_output(self.bin_path+cmd, shell=True)
            if self.clean:
                self.clean_dir()
            print('\nImage successfully converted!')
        except:
            print('Conversion failed. Scan will be ignored.')

        return os.path.join(self.basedir, self.filename)+ext

    def clean_dir(self):

        if os.path.isfile(self.toConvert):
            basedir = self.basedir
        elif os.path.isdir(self.toConvert):
            basedir = self.toConvert
            
        toDelete = glob.glob(basedir+'/*.IMA')
        if not toDelete:
            toDelete = glob.glob(basedir+'/*.dcm')
        if toDelete:
            for f in toDelete:
                os.remove(f)
        else:
            print('No DICOM files to delete found in {}'.format(basedir))
