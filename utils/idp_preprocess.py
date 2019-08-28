import os
import pickle

import numpy as np
from utils.path_utils import radar_data_grapher_volumned, generate_path

specimen_list = [
    # generate_path('zr', 0), generate_path('zr', 1),
    # generate_path('zy', 0), generate_path('zy', 1),
    # generate_path('zy', 2),generate_path('zy', 3)

    # generate_path('zl', 0),generate_path('zl', 1),
    # generate_path('zl', 2),generate_path('zl', 3)
    # generate_path('py', 0),#generate_path('py', 3)
    generate_path('zl', 0), generate_path('zl', 1),
    generate_path('zy', 0), generate_path('zy', 1),
    generate_path('ya', 0), generate_path('ya', 1),
]

# use data augmentation

isDataGen = True

for i, path in enumerate(specimen_list):
    # generate orignial data
    print('Processing specimen #' + str(i) + '__________________________________')
    radar_data_grapher_volumned(path, is_plot=True)
    # radar_data_grapher_volumned(path,  augmentation=['trans'], isDataGen=isDataGen)
    # radar_data_grapher_volumned(path, augmentation=['rot'], isDataGen=isDataGen)
    # radar_data_grapher_volumned(path,  augmentation=['scale'], isDataGen=isDataGen)
    #
    # radar_data_grapher_volumned(path, augmentation=['trans', 'rot'], isDataGen=isDataGen)
    # radar_data_grapher_volumned(path, augmentation=['trans', 'scale'], isDataGen=isDataGen)
    # radar_data_grapher_volumned(path,  augmentation=['rot', 'scale'], isDataGen=isDataGen)
    #
    # radar_data_grapher_volumned(path,  augmentation=['trans', 'rot', 'scale'], isDataGen=isDataGen)
