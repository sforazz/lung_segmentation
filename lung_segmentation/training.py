"Lung segmentation training class"
import os
import math
import logging
import csv
import pickle
from random import sample
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import backend as K
from keras import callbacks as cbks
from lung_segmentation.models import unet_lung
from lung_segmentation.utils import batch_processing
from lung_segmentation.base import LungSegmentationBase
from lung_segmentation.loss import dice_coefficient
from lung_segmentation.dataloader import CSVDataset
from lung_segmentation import transforms as tx
from lung_segmentation.generators import DataLoader
from tensorflow_estimator.python.estimator.canned.linear_testing_utils import AGE_WEIGHT_NAME

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

    def data_split(self, additional_dataset=[], delete_existing=False,
                   test_percentage=0.2):
        "Function to split the whole dataset into training and validation"

        if not os.path.join(self.work_dir, 'image_filemap.csv') or delete_existing:
            self.csv_file = os.path.join(self.work_dir, 'image_filemap.csv')
            w_dirs = [self.work_dir] + additional_dataset
            LOGGER.info('Splitting the dataset into training ({0}%) and validation ({1}%).'
                        .format((100-test_percentage*100), test_percentage*100))
            for directory in w_dirs:
                data = []
                masks = []
                for root, _, files in os.walk(directory):
                    for name in files:
                        if name.endswith('.npy') and 'Raw_data' in name and 'patch' in name:
                            data.append(os.path.join(root, name))
                        elif name.endswith('.npy') and 'Raw_data' not in name and 'patch' in name:
                            masks.append(os.path.join(root, name))

                data = sorted(data)
                masks = sorted(masks)

                x_train, x_test, y_train, y_test = train_test_split(
                    data, masks, test_size=0.2, random_state=42)

                self.x_train = self.x_train + x_train
                self.x_test = self.x_test + x_test
                self.y_train = self.y_train + y_train
                self.y_test = self.y_test + y_test

            images = self.x_train + self.x_test
            masks = self.y_train + self.y_test
            labels = ['train']*len(self.x_train) + ['test']*len(self.x_test)
            data_dict = {}
            data_dict['images'] = images
            data_dict['masks'] = masks
            data_dict['train-test'] = labels

            with open(self.csv_file, 'w') as csvfile:
                writer = csv.writer(csvfile) 
                writer.writerow(data_dict.keys())
                writer.writerows(zip(*data_dict.values()))
        else:
            self.csv_file = os.path.join(self.work_dir, 'image_filemap.csv')

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

#     def prepare_training_parameters(self, n_epochs=100, training_bs=55, validation_bs=55,
#                                      lr_0=2e-4, training_steps=None, validation_steps=None, fold=0,
#                                      weight_name=None, keep_training=True):
# 
#         self.n_epochs = n_epochs
#         self.lr_0 = lr_0
#         self.fold = fold
#         self.weight_name = weight_name
#         self.keep_training = keep_training
#         self.training_bs = training_bs
#         self.validation_bs = validation_bs
# 
#         if training_steps is None:
#             self.training_steps = math.ceil(len(self.x_train)/training_bs)
#             self.validation_steps = math.ceil(len(self.x_test)/validation_bs)
#         else:
#             self.training_steps = training_steps
#             self.validation_steps = validation_steps
# 
#         if keep_training:
#             with open(os.path.join(self.work_dir, 'Training_loss_fold_{}.txt'
#                                    .format(fold)), 'r') as f:
#                 self.all_loss_training = [float(x) for x in f]
#             with open(os.path.join(self.work_dir, 'Validation_loss_fold_{}.txt'
#                                    .format(fold)), 'r') as f:
#                 self.all_loss_val = [float(x) for x in f]
#             self.current_epoch = len(self.all_loss_training)
#         else:
#             self.all_loss_training = []
#             self.all_loss_val = []
#             self.current_epoch = 0

#     def run_training(self):
#         "Function to run the full training"
# 
#         model = unet_lung()
#         patience = 0
#         for e in range(self.current_epoch, self.n_epochs):
#             LOGGER.info('Epoch {}'.format(str(e+1)))
#             if e > 0 or self.transfer_learning or self.keep_training:
#                 model = unet_lung(pretrained_weights=self.weight_name)
#             if self.transfer_learning:
#                 for layer in model.layers[:26]:
#                     layer.trainable=False
#             lr = self.lr_0 * 0.99**e
#             model.compile(optimizer=Adam(lr), loss='binary_crossentropy',
#                           metrics=[dice_coefficient])
#             training_loss = []
#             training_jd = []
#             validation_loss = []
#             validation_jd = []
#             vs = 0
# 
#             LOGGER.info('Training and validation started...')
#             for ts in range(self.training_steps):
#                 print('Training batch {0}/{1}'.format(ts+1, self.training_steps), end="\r")
# 
#                 hist = self.run_batch_all(model, self.x_train, self.y_train, ts,
#                                           self.training_bs)
#                 training_loss.append(hist[0])
#                 training_jd.append(hist[1])
#             for vs in range(self.validation_steps):
#                 print('Validation batch {0}/{1}'.format(vs+1, self.validation_steps), end="\r")
#                 hist = self.run_batch_val_all(model, self.x_test, self.y_test,
#                                               vs, self.validation_bs)
#                 validation_loss.append(hist[0])
#                 validation_jd.append(hist[1])
# 
#             self.all_loss_training.append(np.mean(training_loss))
#             self.all_loss_val.append(np.mean(validation_loss))
#             LOGGER.info('Training and validation for epoch {} ended!'.format(str(e+1)))
#             LOGGER.info('Training loss: {0}. Dice score: {1}'
#                         .format(np.mean(training_loss), np.mean(training_jd)))
#             LOGGER.info('Validation loss: {0}. Dice score: {1}'
#                         .format(np.mean(validation_loss), np.mean(validation_jd)))
#             weight_name = os.path.join(
#                 self.work_dir, 'double_feat_per_layer_BCE_fold_{0}_low_res_mice_da_2.h5'.format(self.fold))
# 
#             if e == 0:
#                 LOGGER.info('Saving network weights...')
#                 model.save_weights(weight_name)
#             elif (e > 0 and (self.all_loss_val[e] < np.min(self.all_loss_val[:-1]))
#                     or (self.all_loss_training[e] < np.min(self.all_loss_training[:-1]))):
#                 patience = 0
#                 LOGGER.info('Saving network weights...')
#                 model.save_weights(weight_name)
#             elif (e >= 0 and ((self.all_loss_val[e] >= np.min(self.all_loss_val[:-1]))
#                     and (self.all_loss_training[e] >= np.min(self.all_loss_training[:-1]))) and patience < 10):
#                 LOGGER.info('No validation loss improvement with respect to '
#                             'the previous epochs. Weights will not be saved.')
#             elif (e >= 0 and ((self.all_loss_val[e] >= np.min(self.all_loss_val[:-1]))
#                     or (self.all_loss_training[e] >= np.min(self.all_loss_training[:-1])))
#                     and patience >= 10):
#                 LOGGER.info('No validation loss improvement with respect '
#                             'to the previous epochs in the last 10 iterations.'
#                             'Training will be stopped.')
#                 break
#             patience = patience+1
#             with open(os.path.join(self.work_dir, 'Training_loss_fold_{}.txt'
#                                    .format(self.fold)), 'a') as f:
#                 f.write(str(np.mean(training_loss))+'\n')
#             with open(os.path.join(self.work_dir, 'Validation_loss_fold_{}.txt'
#                                    .format(self.fold)), 'a') as f:
#                 f.write(str(np.mean(validation_loss))+'\n')
#             K.clear_session()

    def run_training_augmented(self, n_epochs=100, training_bs=55, validation_bs=55,
                               lr_0=2e-4, training_steps=None, validation_steps=None,
                               weight_name=None, data_augmentation=True, keep_training=False):
        "Function to run training with data augmentation"

        if training_steps is None:
            training_steps = math.ceil(len(self.x_train)/training_bs)
            validation_steps = math.ceil(len(self.x_test)/validation_bs)
        else:
            training_steps = training_steps
            validation_steps = validation_steps

        if data_augmentation:
            co_tx = tx.Compose([tx.RandomAffine(rotation_range=(-15,15),
                                                translation_range=(0.1,0.1),
                                                shear_range=(-10,10),
                                                zoom_range=(0.85,1.15),
                                                turn_off_frequency=5,
                                                fill_value='min',
                                                target_fill_mode='constant',
                                                target_fill_value='min')])
        else:
            co_tx = None

        dataset = CSVDataset(filepath=self.csv_file,
                             base_path='',
                             input_cols=['images'],
                             target_cols=['masks'],
                             co_transform=co_tx)

        val_data, train_data = dataset.split_by_column('train-test')

        train_loader = DataLoader(train_data, batch_size=training_bs, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=validation_bs, shuffle=True)
        if weight_name is None:
            weight_name = os.path.join(
                self.work_dir, 'double_feat_per_layer_BCE_augmented.h5')
        # create model
        if self.transfer_learning or keep_training:
            model = unet_lung(pretrained_weights=weight_name)
        else:
            model = unet_lung()
            initial_epoch = 0

        if keep_training:
            with open(os.path.join(self.work_dir, 'training_history.p'), 'rb') as file_pi:
                past_hist = pickle.load(file_pi)
            initial_epoch = len(past_hist['val_loss'])
            lr_0 = past_hist['lr'][-1]

        if self.transfer_learning:
            for layer in model.layers[:26]:
                layer.trainable=False

        model.compile(optimizer=Adam(lr_0), loss='binary_crossentropy',
                      metrics=[dice_coefficient])

        callbacks = [cbks.ModelCheckpoint(weight_name, monitor='val_loss',
                                          save_best_only=True),
                     cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)]

        history = model.fit_generator(
            generator=iter(train_loader),
            steps_per_epoch=training_steps,
            epochs=n_epochs, verbose=1, callbacks=callbacks,
            shuffle=True,
            validation_data=iter(val_loader),
            validation_steps=validation_steps,
            class_weight=None, max_queue_size=10,
            workers=1, use_multiprocessing=False, initial_epoch=initial_epoch)
        if keep_training:
            for key_val in past_hist.keys():
                history.history[key_val] =  past_hist[key_val] + history.history[key_val]
        with open(os.path.join(self.work_dir, 'training_history.p'), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

#     def save2csv(self):
#         "Function to create a csv file before running the training with data augmentation"
#         images = self.x_train + self.x_test
#         masks = self.y_train + self.y_test
#         labels = ['train']*len(self.x_train) + ['test']*len(self.x_test)
#         data_dict = {}
#         data_dict['images'] = images
#         data_dict['masks'] = masks
#         data_dict['train-test'] = labels
#         self.csv_file = os.path.join(
#             self.work_dir, 'image_filemap.csv')
#         with open(self.csv_file, 'w') as csvfile:
#             writer = csv.writer(csvfile) 
#             writer.writerow(data_dict.keys())
#             writer.writerows(zip(*data_dict.values()))
