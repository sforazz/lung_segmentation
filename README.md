# Introduction
This repository contains a deep learning application to segment lungs from CT images. The Convolutional Neural Network used to this task has been trained on 1500 mouse images (with different level of lung fibrosis) acquired with clinical CT, achieving very good results in 2 independent test sets (median [Dice score](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) above 0.96 and median [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance) below 2 mm). It has been applied to segment lungs from high resolution mouse CT images and human CTs as well after re-training using a transfer learning approach (on 35 mice and 9 subjects, respectively).
However, the application provided here is only for clinical CT mouse lung segmentation, but it will be expanded soon to include the other 2 modalities.

# Installation
Currently, this application is only supported for Linux (Ubuntu) operative system, because it needs some executables that are build on this OS.
To install it, follow the following steps:
- Clone or download the repository.
- Open a terminal and cd into it (`cd lung_segmentation`, if the repository is in the current directory)
- (Optional) create a python virtual environment (for example `virtualenv venv`). For more information about virtualenv please refer to [this web page](https://virtualenv.pypa.io/en/latest/)
- (Optional) activate the just created virtualenv `source venv/bin/activate`
- Open the setup.py script and uncomment the version of tensorflow you want to use (cpu or gpu, based on your system). Then save and close the file.
- Type: `python setup.py install`
- cd into the scripts folder (`cd scripts`). Please make sure that your virtualenv is still active when you changed directory.

# Usage
In order to run the lung segmentation application you then just need to type:
```
python run_segmentation.py --input_dir path/to/dicom_data --work_dir path/to/store/results
```
The application will try to automatically download the pre-trained network weights and all the binary files it needs. If something goes wrong you will have to manually download them from [here](insert link) and store them in the repository folder.
By default, the application will run a 5-folds cross validation inference using 5 different weights files and at the end it will take the mean of each prediction in order to provide the best segmentation.
This application has been built to automatically crop the individual DICOM folder (CT_1 in the example below) in order to have one mouse per image. So if you acquired your clinical mouse CT data in batches of more than 1 mouse this application should take care of it automatically. If you have one mouse per image, the cropping will simply remove part of the background.
All the log files will be stored in the `logs` directory. If something went wrong, you should find more information there.

# Input data structure
The structure of the `--input_dir` needs to be as described below:
```
input_dir
├── CT_1
│   ├── dicom_data_1.IMA
│   ├── dicom_data_2.IMA
    .
    .
    .
├── CT_2
│   ├── dicom_data_1.IMA
│   ├── dicom_data_2.IMA
    .
    .
    .
```
To see an example please download the test data from [here](insert link)

# Results
Right now, you will get a folder for each sub-folder in the input directory (CT_1, CT_2 in the structure above). Each folder will contain one NRRD file per mouse with the corresponding lung mask. For example, if there were 4 mice in CT_1, you will get folder called 'CT_1' in the `--work_dir` containing 4 NRRD files (named basename_cropped_mouse_\*.nrrd) with the cropped mouse data, and 4 NRRD files with the corresponding segmented lungs (named basename_cropped_mouse_\*_lung_seg.nrrd). There will be other files with information about the coordinates used for cropping. Those are planned to be used to convert the individual NRRD segmentation back to DICOM but this functionality is not implemented yet. They can be ignored for now.
