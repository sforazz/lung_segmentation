# Introduction
This repository contains a deep learning application to segment lungs from CT images. The Convolutional Neural Network used to this task has been trained on 1500 mouse images (with different level of lung fibrosis) acquired with clinical CT, achieving very good results in 2 independet test sets (median [Dice score](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) above 0.98 and median [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance) below 1 mm). It has also been tested in micro-CT mouse images where, without any additional training, it got a Dice score of 0.83. Finally, it has been applied to human lung segmentation as well after re-training using a transfer learning approach (on 10 subjects).
However, the application provided here is only for clinical CT mouse lung segmentation, but it will be expanded soon to include the other 2 modalities.

# Installation
Currently, this application is only supported for Linux (Ubuntu) operative system, because it needs some executable that are build on this OP.
To install it, follow the following steps:
- Clone or download the repository.
- Open a terminal and cd into it (`cd lung_segmentation`, if the repository is in the current directory)
- (Optional) create a python virtual environment (for example `virtualenv venv`). For more information about virtualenv please refer to [this web page](https://virtualenv.pypa.io/en/latest/)
- (Optional) activate the just created virtualenv `source venv/bin/activate`
- Type: `python setup.py install`
- cd into the scripts folder (`cd scripts`). Please make sure that your virtualenv is still active when you changed directory.

# Usage
In order to run the lung segmetation application you then just need to type:
```
python run_segmentation.py --input_dir path/to/dicom_data --working_dir path/to/store/results
```
The application will try to automatically download the pre-trained network weights and all the binary files it needs. If something goes wrong you will have to manually download them from [here](insert link) and store them in the repository folder.
By default, the application will run a 5-folds cross validation inference using 5 different weights files and at the end it will take the mean of each prediction in order to provide the best segmentation.
This application has been built to automatically crop the individual DICOM folder (mouse_1 in the example above) in order to have one mouse per image. So if you acquired your clinical mouse CT data in batches of more than 1 mouse this application should take care of it automatically. If you have only one mouse per DICOM folder please specify `--no-crop` when you run the `run_segmentation.py`.
All the log files will be stored in the `logs` directory. If something went wrong, you should find more information there.

# Input data structure
The structure of the `--input_dir` needs to be as described below:
```
input_dir
├── mouse_1
│   ├── dicom_data_1.IMA
│   ├── dicom_data_2.IMA
    .
    .
    .
├── mouse_2
│   ├── dicom_data_1.IMA
│   ├── dicom_data_2.IMA
    .
    .
    .
```
To see an example please download the test data from [here](insert link)
