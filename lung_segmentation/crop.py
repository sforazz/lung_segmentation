import os
import nrrd
import numpy as np
from basecore.utils.filemanip import split_filename
import pickle
import nibabel as nib


class ImageCropping():
    
    def __init__(self, image, mask=None, prefix=None):
        print('\nStarting image cropping...')

        self.image = image
        self.mask = mask

        imagePath, imageFilename, imageExt = split_filename(image)
        self.extention = imageExt
        filename = imageFilename.split('.')[0]
        if mask is not None:
            _, maskFilename, maskExt = split_filename(mask)
            maskFilename = maskFilename.replace('.', '_')
            self.maskOutname = os.path.join(imagePath, maskFilename+'_cropped')+maskExt

        if prefix is None and mask is not None:
            self.imageOutname = os.path.join(imagePath, filename+'_cropped')+imageExt
        elif prefix is None and mask is None:
            self.imageOutname = os.path.join(imagePath, filename+'_cropped')
        elif prefix is not None and mask is None:
            self.imageOutname = os.path.join(imagePath, prefix+'_cropped')
        elif prefix is not None and mask is not None:
            self.imageOutname = os.path.join(imagePath, prefix+'_cropped')+imageExt

    def crop_with_mask(self):

        maskData, maskHD = nrrd.read(self.mask)
        if self.extention == '.nrrd':
            imageData, imageHD = nrrd.read(self.image)
    
            space_x = np.abs(imageHD['space directions'][0, 0])
            space_y = np.abs(imageHD['space directions'][1, 1])
            space_z = np.abs(imageHD['space directions'][2, 2])
        elif self.extention == '.nii.gz':
#             maskData = nib.load(self.mask).get_data()
#             maskHD = nib.load(self.mask).header()
            imageData = nib.load(self.image).get_data()
            imageHD = nib.load(self.image).header

            space_x, space_y, space_z = imageHD.get_zooms()

        delta_x = int(10 / space_x)
        delta_y = int(10 / space_y)
        delta_z = int(10 / space_z)

        x, y, z = np.where(maskData==1)

        maskMax = np.max(maskData)
        maskMin = np.min(maskData)
        if maskMax > 1 and maskMin < 0:
            print('This image {} is probably not a mask, as it is not binary. '
                  'It will be ignored. Please check if it is true.'.format(self.mask))
            self.imageOutname = None
            self.maskOutname = None
        else:
            new_x = [np.min(x)-delta_x, np.max(x)+delta_x]
            new_x[0] = 0 if new_x[0] < 0 else new_x[0]
            new_x[1] = imageData.shape[0] if new_x[1] > imageData.shape[0] else new_x[1]
            
            new_y = [np.min(y)-delta_y, np.max(y)+delta_y]
            new_y[0] = 0 if new_y[0] < 0 else new_y[0]
            new_y[1] = imageData.shape[1] if new_y[1] > imageData.shape[1] else new_y[1]
            
            new_z = [np.min(z)-delta_z, np.max(z)+delta_z]
            new_z[0] = 0 if new_z[0] < 0 else new_z[0]
            new_z[1] = imageData.shape[2] if new_z[1] > imageData.shape[2] else new_z[1]
            
#             if ((np.max(x) - np.min(x)) + delta_x*2) > imageData.shape[0]:
#                 new_x = [0, imageData.shape[0]]
#             else:
#                 new_x = [np.min(x)-delta_x, np.max(x)+delta_x]
#             if ((np.max(y) - np.min(y)) + delta_y*2) > imageData.shape[0]:
#                 new_y = [0, imageData.shape[1]]
#             else:
#                 new_y = [np.min(y)-delta_y, np.max(y)+delta_y]
#             if ((np.max(z) - np.min(z)) + delta_z*2) > imageData.shape[0]:
#                 new_z = [0, imageData.shape[2]]
#             else:
#                 new_z = [np.min(z)-delta_z, np.max(z)+delta_z]
#             new_y = [np.min(y)-20, np.max(y)+20]
#             new_z = [np.min(z)-20, np.max(z)+20]
            croppedMask = maskData[new_x[0]:new_x[1], new_y[0]:new_y[1],
                                   new_z[0]:new_z[1]]
            
            croppedImage = imageData[new_x[0]:new_x[1], new_y[0]:new_y[1],
                                     new_z[0]:new_z[1]]
            if self.extention == '.nrrd':
                imageHD['sizes'] = np.array(croppedImage.shape)
                nrrd.write(self.imageOutname, croppedImage, header=imageHD)
            elif self.extention == '.nii.gz':
                im2save = nib.Nifti1Image(croppedImage, affine=nib.load(self.image).affine)
#                 mask2save = nib.Nifti1Image(croppedMask, affine=nib.load(self.mask).affine)
                nib.save(im2save, self.imageOutname)
#                 nib.save(self.maskOutname, mask2save)
            maskHD['sizes'] = np.array(croppedMask.shape)
            nrrd.write(self.maskOutname, croppedMask, header=maskHD)

        print('Cropping done!\n')
        return self.imageOutname, self.maskOutname

    def crop_wo_mask(self):

        im, imageHD = nrrd.read(self.image)
        space_x = np.abs(imageHD['space directions'][0, 0])
        space_y = np.abs(imageHD['space directions'][1, 1])
        space_z = np.abs(imageHD['space directions'][2, 2])
        
        indY = None

        min_size_x = int(20 / space_x)
        if min_size_x > im.shape[0]:
            min_size_x = im.shape[0]
        min_size_y = int(30 / space_y)
        if min_size_y > im.shape[1]:
            min_size_y = im.shape[1]
            indY = im.shape[1]
        min_size_z = int(30 / space_z)
        if min_size_z > im.shape[2]:
            min_size_z = im.shape[2]

        _, _, dimZ = im.shape

        mean_Z = int(np.ceil((dimZ)/2))

        im[im<np.min(im)+824] = np.min(im)

        _, y1 = np.where(im[:, :, mean_Z] != np.min(im))
        im = im[:, np.min(y1)-10:np.min(y1)+min_size_y+10, :]
        x, y = np.where(im[:, :, mean_Z]!=np.min(im))
        if indY is None:
            indY = np.max(y) + np.min(y1)
        uniq = list(set(x))
        xx = [uniq[0]]
        for i in range(1, len(uniq)): 
            if uniq[i]!=uniq[i-1]+1:
                xx.append(uniq[i-1])
                xx.append(uniq[i])
        xx.append(uniq[-1])
        xx = sorted(list(set(xx)))

        im, _ = nrrd.read(self.image)
        n_mice = 0
        out = []
        to_remove = []
        to_add = []
        z_0 = 0
        z_1 = 1
        while z_1 < len(xx):
            size = xx[z_1] - xx[z_0]
            if size < min_size_x:
                to_remove.append(xx[z_1])
                z_1 += 1
            elif size > 2*min_size_x:
                mp = int(size/2)
                to_add.append(xx[z_0]+mp-3)
                to_add.append(xx[z_0]+mp+3)
                z_0 = z_1+1
                z_1 += 2
            else:
                z_0 = z_1+1
                z_1 += 2
        
        for el in to_remove:
            xx.remove(el)
        
        for el in to_add:
            xx.append(el)
        
        xx = sorted(xx)

        if len(xx) % 2 != 0:
            xx.remove(xx[-1])
        
        for i in range(0, len(xx), 2):
            coordinates = {}
            mp = int((xx[i+1] + xx[i])/2)
            croppedImage = im[xx[i]:xx[i+1], indY-int(min_size_y):indY, mean_Z-int(min_size_z/2):mean_Z+int(min_size_z/2)]
            imageHD['sizes'] = np.array(croppedImage.shape)
            coordinates['x'] = [mp-int(min_size_x/2), mp+int(min_size_x/2)]
            coordinates['y'] = [indY-min_size_y, indY]
            coordinates['z'] = [mean_Z-int(min_size_z/2), mean_Z+int(min_size_z/2)]

            with open(self.imageOutname+'_mouse_{}.p'.format(n_mice), 'wb') as fp:
                pickle.dump(coordinates, fp, protocol=pickle.HIGHEST_PROTOCOL)

            nrrd.write(self.imageOutname+'_mouse_{}.nrrd'.format(n_mice),
                       croppedImage, header=imageHD)
            out.append(self.imageOutname+'_mouse_{}.nrrd'.format(n_mice))
            n_mice += 1

        print('Cropping done!\n')
        return out
