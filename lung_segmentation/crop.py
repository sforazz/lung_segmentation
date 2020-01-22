"""
Class to crop CT images to have only one subject per image.
It should work for pre-clinical and clinical images with different
resolutions.
"""
import os
import logging
import pickle
import numpy as np
import nibabel as nib
import nrrd
import cv2
from lung_segmentation.utils import split_filename
import matplotlib.pyplot as plot
from scipy.ndimage.interpolation import rotate
from scipy import ndimage
from skimage.measure import label, regionprops


LOGGER = logging.getLogger('lungs_segmentation')
MOUSE_NAMES = ['mouse_01', 'mouse_02', 'mouse_03',
               'mouse_04', 'mouse_05', 'mouse_06']


class ImageCropping():

    def __init__(self, image, mask=None, prefix=None):
        LOGGER.info('Starting image cropping...')

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
            LOGGER.info('This image {} is probably not a mask, as it is not binary. '
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

            croppedMask = maskData[new_x[0]:new_x[1], new_y[0]:new_y[1],
                                   new_z[0]:new_z[1]]

            croppedImage = imageData[new_x[0]:new_x[1], new_y[0]:new_y[1],
                                     new_z[0]:new_z[1]]
            if self.extention == '.nrrd':
                imageHD['sizes'] = np.array(croppedImage.shape)
                nrrd.write(self.imageOutname, croppedImage, header=imageHD)
            elif self.extention == '.nii.gz':
                im2save = nib.Nifti1Image(croppedImage, affine=nib.load(self.image).affine)
                nib.save(im2save, self.imageOutname)
            maskHD['sizes'] = np.array(croppedMask.shape)
            nrrd.write(self.maskOutname, croppedMask, header=maskHD)

        LOGGER.info('Cropping done!')
        return self.imageOutname, self.maskOutname

    def crop_wo_mask(self, accurate_naming=True):
        """
        Function to crop CT images automatically. It will look for edges
        in the middle slice and will crop the image accordingly.
        If accurate_naming is enabled, the numbering of the cropped
        images will account for missing subjects within the image.
        This will enable you to keep track of mice in longitudinal studies.
        This is for mouse experiment where more than one mouse is acquired
        in one image. If you are not cropping pre-clinical images or you
        are not interested in keep track of the mice across time-points,
        set this to False.
        """

        im, imageHD = nrrd.read(self.image)
        space_x = np.abs(imageHD['space directions'][0, 0])
        space_y = np.abs(imageHD['space directions'][1, 1])
        space_z = np.abs(imageHD['space directions'][2, 2])
        origin_x = imageHD['space origin'][0]
        process = True
        indY = None
        out = []

        min_first_edge = int(80 / space_x)
        min_last_edge = im.shape[0] - int(80 / space_x)

        min_size_x = int(17 / space_x)
        if min_size_x > im.shape[0]:
            min_size_x = im.shape[0]
        min_size_y = int(30 / space_y)
        if min_size_y > im.shape[1]:
            min_size_y = im.shape[1]
            indY = im.shape[1]
        min_size_z = int(45 / space_z)
        if min_size_z > im.shape[2]:
            min_size_z = im.shape[2]

        _, _, dimZ = im.shape

        mean_Z = int(np.ceil((dimZ)/2))
        n_mice_detected = []
        not_correct = True
        angle = 0
        counter = 0
        while not_correct:
            im[im<np.min(im)+824] = np.min(im)
            im[im == 0] = np.min(im)
            for offset in [20, 10, 0, -10, -20]:
                _, y1 = np.where(im[:, :, mean_Z+offset] != np.min(im))
                im[:, np.min(y1)+min_size_y+10:, mean_Z+offset] = 0
                img2, _, _ = self.find_cluster(im[:, :, mean_Z+offset], space_x)
                labels = label(img2)
                regions = regionprops(labels)
                if regions:
                    n_mice_detected.append(len(regions))
                    if offset == 0:
                        xx = [x for y in [[x.bbox[0], x.bbox[2]] for x in regions] for x in y]
                        yy = [x for y in [[x.bbox[1], x.bbox[3]] for x in regions] for x in y]
                else:
                    n_mice_detected.append(0)
            if len(set(n_mice_detected)) == 1 or (len(set(n_mice_detected)) == 2 and 0 in set(n_mice_detected)):
                not_correct = False
            elif counter < 8:
                angle = angle - 2
                LOGGER.warning('Different number of mice have been detected going from down-up '
                               'in the image. This might be due to an oblique orientation '
                               'of the mouse trail. The CT image will be rotated about the z '
                               'direction of %f degrees', np.abs(angle))
                n_mice_detected = []
                indY = None
                im, _ = nrrd.read(self.image)
                im = rotate(im, angle, (0, 2), reshape=False, order=0)
                counter += 1
                if counter % 2 == 0:
                    mean_Z = mean_Z - 10
            else:
                LOGGER.warning('CT image has been rotated of 14° but the number of mice detected '
                               'is still not the same going from down to up. This CT cannot be '
                               'cropped properly and will be excluded.')
                process = False
                not_correct = False

        if process:
            im, _ = nrrd.read(self.image)
            if angle != 0:
                im = rotate(im, angle, (0, 2), reshape=False, order=0)
                im[im == 0] = np.min(im)
            im[im<np.min(im)+824] = np.min(im)
            im[im == 0] = np.min(im)
            im = im[xx[0]:xx[1], yy[0]:yy[1], :]
            hole_size = []
            for z in range(im.shape[2]):
                _, _, zeros = self.find_cluster(im[:, :, z], space_x)
                hole_size.append(zeros)
            mean_Z = np.where(np.asarray(hole_size)==np.max(hole_size))[0][0]
            im, _ = nrrd.read(self.image)
            if angle != 0:
                im = rotate(im, angle, (0, 2), reshape=False, order=0)
                im[im == 0] = np.min(im)
            im[im<np.min(im)+824] = np.min(im)
            im[im == 0] = np.min(im)

            _, y1 = np.where(im[:, :, mean_Z] != np.min(im))
            im[:, np.min(y1)+min_size_y+10:, mean_Z] = 0
            img2, _, _ = self.find_cluster(im[:, :, mean_Z], space_x)
            labels = label(img2)
            regions = regionprops(labels)
            xx = [x for y in [[x.bbox[0], x.bbox[2]] for x in regions] for x in y]
            yy = [x for y in [[x.bbox[1], x.bbox[3]] for x in regions] for x in y]

            im, _ = nrrd.read(self.image)
            if angle != 0:
                im = rotate(im, angle, (0, 2), reshape=False, order=0)
                im[im == 0] = np.min(im)

            average_mouse_size = int(np.round(np.mean([xx[i+1]-xx[i] for i in range(0, len(xx), 2)])))
            fov_mm = space_x*im.shape[0]
            average_hole_size = average_mouse_size // 2
            max_fov = (average_mouse_size + average_hole_size)*5 + average_mouse_size
            max_fov_mm = max_fov*space_x
            fov_diff_mm = (fov_mm - max_fov_mm)/2
#             fov_shift = int(np.round((origin_x - (fov_mm/2))/space_x))

            if fov_diff_mm <= 0:
                LOGGER.warning('The FOV size seems too small to accomodate six mice. This might mean '
                               'that the CT image was not acquired based on a 6-mice batch. For this reasong, '
                               'the accurate naming, if selected, will be turned off, since it is based on '
                               'the assumption that the CT image was acquired with a FOV big enough for 6 mice.')
                accurate_naming = False
            if accurate_naming:
                image_names = MOUSE_NAMES.copy()
                first_edge = xx[0]
                last_edge = xx[-1]
                names2remove = []
                hole_found = 0
                missing_at_edge = False
                min_first_edge = int(np.round(fov_diff_mm/space_x))
                min_last_edge = min_first_edge + max_fov
                min_size_x = average_mouse_size
                if int(len(xx)/2) < 6:
                    LOGGER.info('Less than 6 mice detected, I will try to rename them correctly.')
                    if first_edge > min_first_edge:
                        missing_left = int(np.round((first_edge-min_first_edge)/(min_size_x+average_hole_size)))
                        if missing_left > 0:
                            LOGGER.info('There are {0} voxels between the left margin of the '
                                        'image and the first detected edge. This usually means that '
                                        'there are {1} missing mice on the left-end side. '
                                        'The mouse naming will be updated accordingly.'
                                        .format(first_edge-min_first_edge, missing_left))
                            for m in range(missing_left):
                                names2remove.append(image_names[m])
                            hole_found = hole_found+missing_left
                            missing_at_edge = True
                    if last_edge < min_last_edge:
                        missing_right = int(np.round((min_last_edge-last_edge)/(min_size_x+average_hole_size)))
                        if missing_right > 0:
                            LOGGER.info('There are {0} voxels between the right margin of the '
                                        'image and the last detected edge. This usually means that '
                                        'there are {1} missing mice on the right-end side. '
                                        'The mouse naming will be updated accordingly.'
                                        .format(min_last_edge-last_edge, missing_right))
                            for m in range(missing_right):
                                names2remove.append(image_names[-1-m])
                            hole_found = hole_found+missing_right
                            missing_at_edge = True
                    for ind in names2remove:
                        image_names.remove(ind)

                    mouse_distances = []

                    for i, ind in enumerate(range(1, len(xx)-1, 2)):
                        mouse_index = image_names[i]
                        distance = xx[ind+1] - xx[ind]
                        mouse_distances.append(distance)
                        hole_dimension = int(np.round(distance/(min_size_x)))
                        if hole_dimension >= 2:
                            names2remove = []
                            LOGGER.info('The distance between mouse {0} and mouse {1} is '
                                        '{4} voxels, which is {2} times greater than the minimum '
                                        'mouse size. This could mean that {3} mice are missing'
                                        ' in this batch. They will'
                                        ' be ignored and the naming will be updated accordingly.'
                                        .format(mouse_index, image_names[i+1], hole_dimension,
                                                hole_dimension-1, distance))
                            for m in range(hole_dimension-1):
                                names2remove.append(image_names[i+m+1])
                            for ind in names2remove:
                                image_names.remove(ind)
                            hole_found += (hole_dimension-1)
                    if hole_found + int(len(xx)/2) < 6:
                        names2remove = []
                        still_missing = 6 - (hole_found + int(len(xx)/2))
                        LOGGER.warning('It seems that not all holes has been identified, since the '
                                    'detected mice are {0} and the hole detected are {1}. '
                                    'This means that there are still {2} mice missing in order to '
                                    'reach the standard mice number (6). I will remove the names '
                                    'belonging to the mouse with the biggest distance.'
                                    .format(int(len(xx)/2), hole_found, still_missing))
                        for i in range(still_missing):
                            max_distance = np.where(np.asarray(mouse_distances)==
                                                    np.max(np.asarray(mouse_distances)))[0][0]
                            names2remove.append(image_names[max_distance+1])
                            mouse_distances[max_distance] = 0
                        for ind in names2remove:
                                image_names.remove(ind)
                    elif hole_found + int(len(xx)/2) > 6:
                        LOGGER.warning('The accurate naming failed because the algorithm detected too many '
                                    'missing mice. For this reason the accurate naming will be swithed off.')
                        image_names = ['mouse_0{}'.format(x+1) for x in range(int(len(xx)//2))]
            else:
                image_names = ['mouse_0{}'.format(x+1) for x in range(int(len(xx)//2))]

            offset_box = average_hole_size // 3
            y_min = np.min(yy) - offset_box
            y_max = np.max(yy) + offset_box
            for n_mice, i in enumerate(range(0, len(xx), 2)):
                coordinates = {}
                croppedImage = im[xx[i]-offset_box:xx[i+1]+offset_box, y_min:y_max,
                                  mean_Z-int(min_size_z/2):mean_Z+int(min_size_z/2)]
                imageHD['sizes'] = np.array(croppedImage.shape)
                coordinates['x'] = [xx[i]-offset_box, xx[i]+offset_box]
                coordinates['y'] = [y_min, y_max]
                coordinates['z'] = [mean_Z-int(min_size_z/2), mean_Z+int(min_size_z/2)]

                with open(self.imageOutname+'_{}.p'.format(image_names[n_mice]), 'wb') as fp:
                    pickle.dump(coordinates, fp, protocol=pickle.HIGHEST_PROTOCOL)

                nrrd.write(self.imageOutname+'_{}.nrrd'.format(image_names[n_mice]),
                           croppedImage, header=imageHD)
                out.append(self.imageOutname+'_{}.nrrd'.format(image_names[n_mice]))

        LOGGER.info('Cropping done!')
        return out

    def find_cluster(self, im, spacing):

        im[im==np.min(im)] = 0
        im[im!=0] = 1

        nb_components, output, stats, _ = (
            cv2.connectedComponentsWithStats(im.astype(np.uint8),
                                             connectivity=8))
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        min_size = 100/spacing
        img2 = np.zeros((output.shape))
        cluster_size = []
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                cluster_size.append(sizes[i])
                img2[output == i + 1] = 1
        img2_filled = ndimage.binary_fill_holes(img2)
        zeros = np.sum(img2_filled-img2)

        return img2, cluster_size, zeros
    