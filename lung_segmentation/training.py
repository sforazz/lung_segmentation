from lung_segmentation.base import LungSegmentationBase
import os
from lung_segmentation.utils import batch_processing
import logging
import random
from sklearn.model_selection import train_test_split
from lung_segmentation.models import unet_lung
from dl.losses.dice import dice_coefficient, loss_dice_coefficient_error
from keras.optimizers import Adam
from random import sample
import math
import numpy as np
from keras import backend as K


logger = logging.getLogger('lungs_segmentation')


class LungSegmentationTraining(LungSegmentationBase):
    
    def get_data(self, root_path='', testing=True):

        self.precomputed_masks = []
        self.precomputed_images = []
        testing_dir = os.path.join(self.work_dir, 'testing')
        self.work_dir = os.path.join(self.work_dir, 'training')
        if not os.path.isdir(self.work_dir):
            os.mkdir(self.work_dir)

        self.dcm_folders, self.mask_paths = batch_processing(self.input_path, root=root_path)

        if testing:
            if not os.path.isdir(testing_dir):
                os.mkdir(testing_dir)
            if os.path.isfile(os.path.join(testing_dir, 'test_subjects.txt')):
                logger.info('Found a text file with the subjects to use for testing')
                with open(os.path.join(testing_dir, 'test_subjects.txt'), 'r') as f:
                    self.test_set = [x.strip() for x in f]
            else:
                len_test_set = int(len(self.dcm_folders)*0.1) if int(len(self.dcm_folders)*0.1) > 0 else 1
                test_indexes = random.sample(range(len(self.dcm_folders)), len_test_set)
                self.test_set = [self.dcm_folders[x] for x in test_indexes]
                logger.info('{} folders have been removed from the dataset to use '
                            'them as testing cohort.'.format(len(self.test_set)))
                with open(os.path.join(testing_dir, 'test_subjects.txt'), 'w') as f:
                    for s in self.test_set:
                        f.write(s+'\n')
#         self.dcm_folders = [x for x in self.dcm_folders if x not in self.test_set]
        
#         if os.path.isfile(os.path.join(self.work_dir, 'processed_DICOM.txt')):
#             with open(os.path.join(self.work_dir, 'processed_DICOM.txt'), 'r') as f:
#                 self.processed_subs = [x.strip() for x in f]
#             logger.info('Found {} already processed subjects. They will be skipped '
#                         'from the preprocessing.'.format(len(self.processed_subs)))
# 
#         if os.path.isfile(os.path.join(self.work_dir, 'processed_NRRD.txt')):
#             with open(os.path.join(self.work_dir, 'processed_NRRD.txt'), 'r') as f:
#                 processed_nrrd = [x.strip() for x in f]
#             for sub in processed_nrrd:
#                 self.precomputed_images = self.precomputed_images + sorted(glob.glob(
#                     os.path.join(sub, 'Raw_data*resampled.nrrd')))
#                 masks = [x for x in sorted(glob.glob(os.path.join(sub, '*resampled.nrrd'))) if 'Raw_data' not in x]
#                 self.precomputed_masks = self.precomputed_masks + masks

#         if os.path.isfile(os.path.join(self.work_dir, 'test_subjects.txt')):
#             logger.info('Found a text file with the subjects to use for testing')
#             with open(os.path.join(self.work_dir, 'test_subjects.txt'), 'r') as f:
#                 self.test_set = [x.strip() for x in f]
#         else:
#             self.test_set = None
            
#         self.dcm_folders = [x for x in raw_data if x not in self.processed_subs and x not in self.test_set]
#         unproceseed_indexes = [i for i, x in enumerate(raw_data) if x in self.dcm_folders]
#         self.mask_paths = [self.mask_paths[i] for i in unproceseed_indexes]
        logger.info('{} folders will be pre-processed and use to train the network.'
                    .format(len(self.dcm_folders)))
    
    def create_tensors(self, patch_size=(96, 96), save2npy=True):
        return LungSegmentationBase.create_tensors(self, patch_size=patch_size, save2npy=save2npy)
    
    def data_split(self):
        
        data = []
        masks = []
        logger.info('Splitting the dataset into training (70%) and validation (30%).')
        for root, _, files in os.walk(self.work_dir): 
            for name in files: 
                if name.endswith('.npy') and 'Raw_data' in name and 'patch' in name: 
                    data.append(os.path.join(root, name))
                elif name.endswith('.npy') and 'Raw_data' not in name and 'patch' in name: 
                    masks.append(os.path.join(root, name))
    
        data = sorted(data)
        masks = sorted(masks)
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, masks, test_size=0.3, random_state=42)
    
    def run_batch_all(self, model, batch_files, batch_masks, s, batch_size):
        files = batch_files[s*batch_size:(s+1)*batch_size]
        masks = batch_masks[s*batch_size:(s+1)*batch_size]
        x = np.asarray([np.load(x) for x in files]).reshape(-1,96,96,1)
        y = np.asarray([np.load(x) for x in masks]).reshape(-1,96,96,1)
        hist = model.train_on_batch(x, y)
        
        return hist
    
    
    def run_batch_val_all(self, model, batch_files, batch_masks, s, batch_size):
        files = batch_files[s*batch_size:(s+1)*batch_size]
        masks = batch_masks[s*batch_size:(s+1)*batch_size]
        x = np.asarray([np.load(x) for x in files]).reshape(-1,96,96,1)
        y = np.asarray([np.load(x) for x in masks]).reshape(-1,96,96,1)
        hist = model.test_on_batch(x, y)
        
        return hist

    def run_training(self, n_epochs=100, training_bs=41, validation_bs=40,
                     lr_0=2e-4, training_steps=None, validation_steps=None, fold=0):

        if training_steps is None:
            training_steps = math.ceil(len(self.X_train)/training_bs)
            validation_steps = math.ceil(len(self.X_test)/validation_bs)

        model = unet_lung()
        
        all_loss_training = []
        all_loss_val = []
        patience = 0
        weight_name = None
        for e in range(n_epochs):
            logger.info('Epoch {}'.format(str(e+1)))
            if e > 0:
                model = unet_lung(pretrained_weights=weight_name)
            lr = lr_0 * 0.99**e
            model.compile(loss=loss_dice_coefficient_error,
                          optimizer=Adam(lr), metrics=[dice_coefficient])
            training_loss = []
            training_jd = []
            validation_loss = []
            validation_jd = []
            validation_index = sample(range(10, training_steps), validation_steps)
            vs = 0
        
            logger.info('Training and validation started...')
            for ts in range(training_steps):
                print('Batch {0}/{1}'.format(ts+1, training_steps), end="\r")
                hist = self.run_batch_all(model, self.X_train, self.y_train, ts, training_bs)
                training_loss.append(hist[0])
                training_jd.append(hist[1])
                if ts in validation_index:
                    hist = self.run_batch_val_all(model, self.X_test, self.y_test, vs, validation_bs)
                    validation_loss.append(hist[0])
                    validation_jd.append(hist[1])
                    vs = vs+1
                    
            all_loss_training.append(np.mean(training_loss))
            all_loss_val.append(np.mean(validation_loss))
            logger.info('Training and validation for epoch {} ended!'.format(str(e+1)))
            logger.info('Training loss: {0}. Dice score: {1}'
                        .format(np.mean(training_loss), np.mean(training_jd)))
            logger.info('Validation loss: {0}. Dice score: {1}'
                        .format(np.mean(validation_loss), np.mean(validation_jd)))
            weight_name = os.path.join(self.work_dir, 
                                       'double_feat_per_layer_cross_ent_fold_{0}.h5'.format(fold))
            if e == 0:
                logger.info('Saving network weights...')
                model.save_weights(weight_name)
            elif (e > 0 and (all_loss_val[e] < np.min(all_loss_val[:-1]))
                    and (all_loss_training[e] < np.min(all_loss_training[:-1]))):
                patience = 0
                logger.info('Saving network weights...')
                model.save_weights(weight_name)
            elif (e >= 0 and ((all_loss_val[e] >= np.min(all_loss_val[:-1]))
                    or (all_loss_training[e] >= np.min(all_loss_training[:-1]))) and patience < 10):
                logger.info('No validation loss improvement with respect to '
                            'the previous epochs. Weights will not be saved.')
            elif (e >= 0 and ((all_loss_val[e] >= np.min(all_loss_val[:-1]))
                    or (all_loss_training[e] >= np.min(all_loss_training[:-1]))) and patience >= 10):
                logger.info('No validation loss improvement with respect '
                            'to the previous epochs in the last 10 iterations.'
                            'Training will be stopped.')
                break
            patience = patience+1
            K.clear_session()
        
        np.savetxt(os.path.join(self.work_dir, 'Training_loss_fold_{}.txt'.format(fold)),
                   np.asarray(all_loss_training))
        np.savetxt(os.path.join(self.work_dir, 'Validation_loss_fold_{}.txt').format(fold),
                   np.asarray(all_loss_val))
    