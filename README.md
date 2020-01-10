# Introduction
This repository contains a deep learning application to segment lungs from CT images. The Convolutional Neural Network used to this task has been trained on 1500 mouse images (with different level of lung fibrosis) acquired with clinical CT, achieving very good results in 2 independent test sets (median [Dice score](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) above 0.96 and median [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance) below 2 mm). It has been applied to segment lungs from high resolution mouse CT images and human CTs as well after re-training using a transfer learning approach (on 35 mice and 9 subjects, respectively).
The lung segmentation inference can be run through command line or using a GUI.

# Installation
Currently, this application is only supported for Linux (developped and tested on Ubuntu 18.04) operative system, because it needs some executables that are build on this OS. It works with python 3x.
To install it, follow the following steps:
- Clone or download this repository.
- Open a terminal and cd into it (`cd lung_segmentation`, if the repository is in the current directory)
- (Optional) create a python virtual environment (for example `virtualenv venv`). For more information about virtualenv please refer to [this web page](https://virtualenv.pypa.io/en/latest/)
- (Optional) activate the just created virtualenv `source venv/bin/activate`
- By default, the CPU version of Tensorflow will be installed using the setup.py. However, if you have GPU availablility, you can open the setup.py file, comment the line where is specified the tensorflow package and uncomment the one with the GPU version (used during development). Then save and close the file.
- Type: `python setup.py install`
- (Optional, to use the GUI) install tkinter `sudo apt-get install python3-tk`
- (Optional, to use the GUI) type `python gui_setup.py`. This will install the GUI and you will find it in your "Applications" under the name Lung Segmentation.
- If you want to use the command line application, just cd into the scripts folder (`cd scripts`). Please make sure that your virtualenv is still active when you changed directory.

# Command line script usage
For a very basic usage of the lung segmentation application, from command line, you just need to type (inside the "script" folder):
```
python run_inference.py --input_path path/to/dicom_data --work_dir path/to/store/results
```
Where `input_path` can be either a folder with several sub-folders (one per CT image) containing DICOM data, or che be an Excel sheet with one CT image per raw (See "Input data structure" paragraph below for more information). `work_dir` is the path to save the results (if the directory does not exist, it will be created).
The application will try to automatically download the pre-trained network weights and all the binary files it needs. If something goes wrong you will have to manually download them from [here](insert link) and store them in the repository folder (binary files must go into a folder called "bin" and the pre-trained weights in the "weights" folder).
By default, the application will run a 5-folds cross validation inference using 5 different weights files and at the end it will calculate the average prediction in order to provide the best segmentation.
This application has been built to automatically crop the individual DICOM CT image (CT_1 in the example below) in order to have one mouse per image. So if you acquired your clinical mouse CT data in batches of more than 1 mouse this application should take care of it automatically. If you have one mouse per image, the cropping will simply remove part of the background.
All the log files will be stored in the `logs` directory. If something went wrong, you should find more information there.

# Input data structure
1) If you provide a folder in `--input_path` then its structure needs to be as described below:
```
input_path
├── CT_1
│   ├── dicom_data_1.dcm
│   ├── dicom_data_2.dcm
    .
    .
    .
├── CT_2
│   ├── dicom_data_1.dcm
│   ├── dicom_data_2.dcm
    .
    .
    .
```
To see an example please download the test data from [here](insert link)
2) If you provide an Excel sheet in `--input_path` then each raw has to contain the path to a folder contain one CT image. The first raw must contain the word "subjects" in order to be found by the data loader.
Optionally, when you use an Excel sheet as input and ONLY if you have ground truth lung mask (for example you trained the network from scratch and you want to test it), you can provide a second column with the path the folder containing the lung masks corresponding to the CT image. The two columns must have the same number of entries. The mask folder will have as many .nrrd files as the number of subjects (mice or humans) in the corresponding CT image. If for example. there are 4 mice in one CT image, then the mask folder must contain 4 .nrrd lung masks.
# Results
Right now, you will get a folder for each sub-folder in the input directory (CT_1, CT_2 in the structure above). Each folder will contain one NRRD file per mouse with the corresponding lung mask. For example, if there were 4 mice in CT_1, you will get folder called 'CT_1' in the `--work_dir` containing 4 NRRD files (named basename_cropped_mouse_\*.nrrd) with the cropped mouse data, and 4 NRRD files with the corresponding segmented lungs (named basename_cropped_mouse_\*_lung_seg.nrrd). There will be other files with information about the coordinates used for cropping. Those are planned to be used to convert the individual NRRD segmentation back to DICOM but this functionality is not implemented yet. They can be ignored for now.
