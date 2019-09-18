"Lung segmentation training class"
import os
import math
import logging
from random import sample
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import backend as K
from lung_segmentation.models import unet_lung
from lung_segmentation.utils import batch_processing
from lung_segmentation.base import LungSegmentationBase
from lung_segmentation.loss import dice_coefficient


LOGGER = logging.getLogger('lungs_segmentation')


class LungSegmentationTraining(LungSegmentationBase):
    "Class to run the whole training process"
    def get_data(self, root_path='', testing=True, preproc_only=False):
        "Function to get the data for training"
        self.precomputed_masks = []
        self.precomputed_images = []
        self.testing = False
        testing_dir = os.path.join(self.work_dir, 'testing')
        if not preproc_only:
            self.work_dir = os.path.join(self.work_dir, 'training')
        else:
            self.work_dir = os.path.join(self.work_dir, 'pre-processing')
        if not os.path.isdir(self.work_dir):
            os.mkdir(self.work_dir)

        self.dcm_folders, self.mask_paths = batch_processing(self.input_path, root=root_path)

        if testing:
            if not os.path.isdir(testing_dir):
                os.mkdir(testing_dir)
            if os.path.isfile(os.path.join(testing_dir, 'test_subjects.txt')):
                LOGGER.info('Found a text file with the subjects to use for testing')
                with open(os.path.join(testing_dir, 'test_subjects.txt'), 'r') as f:
                    self.test_set = [x.strip() for x in f]
            else:
                len_test_set = (int(len(self.dcm_folders)*0.1)
                                if int(len(self.dcm_folders)*0.1) > 0 else 1)
                test_indexes = sample(range(len(self.dcm_folders)), len_test_set)
                self.test_set = [self.dcm_folders[x] for x in test_indexes]
                test_set_gt = [self.mask_paths[x] for x in test_indexes]
                LOGGER.info('{} folders have been removed from the dataset to use '
                            'them as testing cohort.'.format(len(self.test_set)))
                with open(os.path.join(testing_dir, 'test_subjects.txt'), 'w') as f:
                    for s in self.test_set:
                        f.write(s+'\n')
                with open(os.path.join(testing_dir, 'test_subjects_gt_masks.txt'), 'w') as f:
                    for s in test_set_gt:
                        f.write(s+'\n')

        LOGGER.info('{} folders will be pre-processed and use to train the '
                    'network (if network training was selected).'
                    .format(len(self.dcm_folders)))

    def create_tensors(self, patch_size=(96, 96), save2npy=True):
        "Function to create the tensors used for training the CNN"
        return LungSegmentationBase.create_tensors(self, patch_size=patch_size, save2npy=save2npy)

    def data_split(self, additional_dataset=[]):
        "Function to split the whole dataset into training and validation"
        data = []
        masks = []
        w_dirs = [self.work_dir] + additional_dataset
        LOGGER.info('Splitting the dataset into training (70%) and validation (30%).')
        for directory in w_dirs:
            for root, _, files in os.walk(directory):
                for name in files:
                    if name.endswith('.npy') and 'Raw_data' in name and 'patch' in name:
                        data.append(os.path.join(root, name))
                    elif name.endswith('.npy') and 'Raw_data' not in name and 'patch' in name:
                        masks.append(os.path.join(root, name))

            data = sorted(data)
            masks = sorted(masks)

            x_train, x_test, y_train, y_test = train_test_split(
                data, masks, test_size=0.3, random_state=42)

            self.x_train = self.x_train + x_train
            self.x_test = self.x_test + x_test
            self.y_train = self.y_train + y_train
            self.y_test = self.y_test + y_test

    @staticmethod
    def run_batch_all(model, batch_files, batch_masks, step, batch_size):
        "Method to run training for 1 batch"
        files = batch_files[step*batch_size:(step+1)*batch_size]
        masks = batch_masks[step*batch_size:(step+1)*batch_size]
        images = np.asarray([np.load(x) for x in files]).reshape(-1, 96, 96, 1)
        labels = np.asarray([np.load(x) for x in masks]).reshape(-1, 96, 96, 1)
        hist = model.train_on_batch(images, labels)

        return hist

    @staticmethod
    def run_batch_val_all(model, batch_files, batch_masks, step, batch_size):
        "Method to run validation for 1 batch"
        files = batch_files[step*batch_size:(step+1)*batch_size]
        masks = batch_masks[step*batch_size:(step+1)*batch_size]
        images = np.asarray([np.load(x) for x in files]).reshape(-1, 96, 96, 1)
        labels = np.asarray([np.load(x) for x in masks]).reshape(-1, 96, 96, 1)
        hist = model.test_on_batch(images, labels)

        return hist

    @staticmethod
    def run_batch(model, batch_files, batch_masks, s, batch_size):

        indexes = sample(range(len(batch_files)), batch_size)
        files = [batch_files[x] for x in indexes]
        masks = [batch_masks[x] for x in indexes]
        images = np.asarray([np.load(x) for x in files]).reshape(-1, 96, 96, 1)
        labels = np.asarray([np.load(x) for x in masks]).reshape(-1, 96, 96, 1)
        hist = model.train_on_batch(images, labels)

        return hist

    @staticmethod
    def run_batch_val(model, batch_files, batch_masks, s, batch_size):

        indexes = sample(range(len(batch_files)), batch_size)
        files = [batch_files[x] for x in indexes]
        masks = [batch_masks[x] for x in indexes]
        images = np.asarray([np.load(x) for x in files]).reshape(-1, 96, 96, 1)
        labels = np.asarray([np.load(x) for x in masks]).reshape(-1, 96, 96, 1)
        hist = model.test_on_batch(images, labels)

        return hist

    def run_training(self, n_epochs=100, training_bs=41, validation_bs=40,
                     lr_0=2e-4, training_steps=None, validation_steps=None, fold=0,
                     weight_name=None):
        "Function to run the full training"
        if training_steps is None:
            training_steps = math.ceil(len(self.x_train)/training_bs)
            validation_steps = math.ceil(len(self.x_test)/validation_bs)

        model = unet_lung()

        all_loss_training = []
        all_loss_val = []
        patience = 0
        for e in range(n_epochs):
            LOGGER.info('Epoch {}'.format(str(e+1)))
            if e > 0 or self.transfer_learning:
                model = unet_lung(pretrained_weights=weight_name)
            if self.transfer_learning:
                for layer in model.layers[:26]:
                    layer.trainable=False
            lr = lr_0 * 0.99**e
            model.compile(optimizer=Adam(lr), loss='binary_crossentropy',
                          metrics=[dice_coefficient])
            training_loss = []
            training_jd = []
            validation_loss = []
            validation_jd = []
            validation_index = sample(range(10, training_steps), validation_steps)
            vs = 0

            LOGGER.info('Training and validation started...')
            for ts in range(training_steps):
                print('Batch {0}/{1}'.format(ts+1, training_steps), end="\r")
                hist = self.run_batch_all(model, self.x_train, self.y_train, ts, training_bs)
                training_loss.append(hist[0])
                training_jd.append(hist[1])
                if ts in validation_index:
                    hist = self.run_batch_val_all(model, self.x_test, self.y_test,
                                                  vs, validation_bs)
                    validation_loss.append(hist[0])
                    validation_jd.append(hist[1])
                    vs = vs+1

            all_loss_training.append(np.mean(training_loss))
            all_loss_val.append(np.mean(validation_loss))
            LOGGER.info('Training and validation for epoch {} ended!'.format(str(e+1)))
            LOGGER.info('Training loss: {0}. Dice score: {1}'
                        .format(np.mean(training_loss), np.mean(training_jd)))
            LOGGER.info('Validation loss: {0}. Dice score: {1}'
                        .format(np.mean(validation_loss), np.mean(validation_jd)))
            weight_name = os.path.join(self.work_dir,
                                       'double_feat_per_layer_BCE_fold_{0}_plus_MA_plus_human2.h5'.format(fold))

            if e == 0:
                LOGGER.info('Saving network weights...')
                model.save_weights(weight_name)
            elif (e > 0 and (all_loss_val[e] < np.min(all_loss_val[:-1]))
                    and (all_loss_training[e] < np.min(all_loss_training[:-1]))):
                patience = 0
                LOGGER.info('Saving network weights...')
                model.save_weights(weight_name)
            elif (e >= 0 and ((all_loss_val[e] >= np.min(all_loss_val[:-1]))
                    or (all_loss_training[e] >= np.min(all_loss_training[:-1]))) and patience < 10):
                LOGGER.info('No validation loss improvement with respect to '
                            'the previous epochs. Weights will not be saved.')
            elif (e >= 0 and ((all_loss_val[e] >= np.min(all_loss_val[:-1]))
                    or (all_loss_training[e] >= np.min(all_loss_training[:-1])))
                    and patience >= 10):
                LOGGER.info('No validation loss improvement with respect '
                            'to the previous epochs in the last 10 iterations.'
                            'Training will be stopped.')
                break
            patience = patience+1
            K.clear_session()

        np.savetxt(os.path.join(self.work_dir, 'Training_loss_fold_{}.txt'.format(fold)),
                   np.asarray(all_loss_training))
        np.savetxt(os.path.join(self.work_dir, 'Validation_loss_fold_{}.txt').format(fold),
                   np.asarray(all_loss_val))
        