"Lung segmentation training class"
import os
import math
import logging
import csv
import pickle
from random import sample
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import callbacks as cbks
from lung_segmentation.models import unet_lung
from lung_segmentation.utils import batch_processing
from lung_segmentation.base import LungSegmentationBase
from lung_segmentation.loss import dice_coefficient, loss_dice_coefficient_error, combined_loss
from lung_segmentation.dataloader import CSVDataset
from lung_segmentation import transforms as tx
from lung_segmentation.generators import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import glob


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
                   test_percentage=0.2, fold=5):
        "Function to split the whole dataset into training and validation"
        self.csv_file = sorted(glob.glob(os.path.join(self.work_dir, 'image_filemap_fold*.csv')))
        if len(self.csv_file) != fold or delete_existing:
            for csv_f in self.csv_file:
                os.remove(csv_f)
            self.csv_file = []
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
                    data, masks, test_size=test_percentage, random_state=42)

                self.x_train = self.x_train + x_train
                self.x_test = self.x_test + x_test
                self.y_train = self.y_train + y_train
                self.y_test = self.y_test + y_test

            images = self.x_train + self.x_test
            masks = self.y_train + self.y_test
            data_dict = {}
            data_dict['images'] = images
            data_dict['masks'] = masks
            if fold > 1:
                kf = KFold(n_splits=fold)
                fold_number = 0
                for train_index, test_index in kf.split(images):
                    labels = np.zeros(len(images), dtype='U5')
                    labels[train_index] = 'train'
                    labels[test_index] = 'test'
                    data_dict['train-test'] = labels
                    with open(os.path.join(self.work_dir, 'image_filemap_fold{}.csv'
                                           .format(fold_number)), 'w') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(data_dict.keys())
                        writer.writerows(zip(*data_dict.values()))
                    self.csv_file.append(os.path.join(self.work_dir, 'image_filemap_fold{}.csv'
                                                      .format(fold_number)))
                    fold_number += 1
            else:
                labels = ['train']*len(self.x_train) + ['test']*len(self.x_test)
                data_dict['train-test'] = labels
                with open(os.path.join(self.work_dir, 'image_filemap_fold0.csv'), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(data_dict.keys())
                    writer.writerows(zip(*data_dict.values()))

    def run_training(self, n_epochs=100, training_bs=50, validation_bs=50,
                     lr_0=2e-4, training_steps=None, validation_steps=None,
                     weight_name=None, data_augmentation=True, keep_training=False):
        "Function to run training with data augmentation"
        for n_fold, csv_file in enumerate(self.csv_file):
            LOGGER.info('Running training for fold {}'.format(n_fold+1))
            if data_augmentation:
                co_tx = tx.Compose([tx.RandomAffine(rotation_range=(-35,35),
                                                    translation_range=(0.4,0.4),
                                                    shear_range=(-30,30),
                                                    zoom_range=(0.45,1.55),
                                                    turn_off_frequency=5,
                                                    fill_value='min',
                                                    target_fill_mode='constant',
                                                    target_fill_value='min')])
            else:
                co_tx = None

            dataset = CSVDataset(filepath=csv_file,
                                 base_path='',
                                 input_cols=['images'],
                                 target_cols=['masks'],
                                 co_transform=co_tx)

            val_data, train_data = dataset.split_by_column('train-test')

            if training_steps is None:
                training_steps = math.ceil(len(train_data)/training_bs)
                validation_steps = math.ceil(len(val_data)/validation_bs)
            else:
                training_steps = training_steps
                validation_steps = validation_steps

            train_loader = DataLoader(train_data, batch_size=training_bs, shuffle=False)
            val_loader = DataLoader(val_data, batch_size=validation_bs, shuffle=False)
            if weight_name is None:
                weight_name = os.path.join(
                    self.work_dir, 'double_feat_per_layer_BCE_augmented_fold{}.h5'.format(n_fold+1))

            # create model
            initial_epoch = 0
            if self.transfer_learning or keep_training:
                model = unet_lung(pretrained_weights=weight_name)
            else:
                model = unet_lung()

            if keep_training:
                try:
                    with open(os.path.join(self.work_dir, 'training_history_fold{}.p'
                                           .format(n_fold+1)), 'rb') as file_pi:
                        past_hist = pickle.load(file_pi)
                    initial_epoch = len(past_hist['val_loss'])
                    lr_0 = past_hist['lr'][-1]
                except FileNotFoundError:
                    LOGGER.info('No training history found. The training will start from epoch 1')

            if self.transfer_learning:
                weight_name_0 = weight_name
                for layer in model.layers[:25]:
                    layer.trainable=False
                weight_name = os.path.join(
                    self.work_dir, 'double_feat_per_layer_BCE_augmented_tl_fold{}.h5'
                    .format(n_fold+1))

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
            with open(os.path.join(self.work_dir, 'training_history_fold{}.p'
                                   .format(n_fold+1)), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            if self.transfer_learning:
                weight_name = weight_name_0
            else:
                weight_name = None
