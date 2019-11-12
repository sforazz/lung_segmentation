"Script to setup the GUI in the Applications (for Linux)"
import os


HOME_DIR = os.environ['HOME']
LIBRARY_PATH = os.environ['LD_LIBRARY_PATH']
PYTHONPATH = os.environ['PYTHONPATH']
APP_DIR = os.path.abspath(os.path.split(__file__)[0])
VENV = None

for root, _, files in os.walk(APP_DIR):
    for name in files:
        if name == 'activate':
            VENV = os.path.join(root, name)

LINUX_TEMPLATE = (
"""[Desktop Entry]
Encoding=UTF-8
Version=1.0
Type=Application
Name=Lung Segmentation
Icon={0}/lung_seg.png
Exec={0}/scripts/run_gui.sh
Terminal=true
Categories=Application;""".format(APP_DIR))
with open(os.path.join(HOME_DIR, '.local/share/applications/lung_seg_app.desktop'), 'w') as f:
    f.write(LINUX_TEMPLATE)

GUI_TEMPLATE = (
"""#!/bin/bash
source {0}
export PYTHONPATH={1}:$PYTHONPATH
export LD_LIBRARY_PATH={2}
python '{1}/scripts/run_inference_gui.py'"""
.format(VENV, APP_DIR, LIBRARY_PATH))
with open(os.path.join(APP_DIR, 'scripts', 'run_gui_test.sh'), 'w') as f:
    f.write(GUI_TEMPLATE)
