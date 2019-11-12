"Configuration file"


STANDARD_CONFIG = {
    'spacing': (0.35, 0.35, 0.35),
    'cluster_correction': False,
    'min_extent': 350,
    'dicom_check': True,
    'weights_url': ('https://angiogenesis.dkfz.de/oncoexpress'
                    '/software/delineation/bin/weights.tar.gz')}


HIGH_RES_CONFIG = {
    'spacing': (0.2, 0.2, 0.2),
    'cluster_correction': True,
    'min_extent': 100000,
    'dicom_check': False,
    'weights_url': ('https://angiogenesis.dkfz.de/oncoexpress'
                    '/software/delineation/bin/highres_weights.tar.gz')}


HUMAN_CONFIG = {
    'spacing': (1, 1, 1),
    'cluster_correction': True,
    'min_extent': 300000,
    'dicom_check': False,
    'weights_url': ('https://angiogenesis.dkfz.de/oncoexpress'
                    '/software/delineation/bin/human_weights.tar.gz')}


NONE_CONFIG = {
    'spacing': (0, 0, 0),
    'cluster_correction': False,
    'min_extent': 0,
    'dicom_check': False,
    'weights_url': None}
