import os
import pickle

import numpy as np
from utils.path_utils import idp_preprocess, generate_path, thm_preprocess

specimen_list = [
    generate_path('py', 0, mode='thm'), generate_path('py', 1, mode='thm'),
    generate_path('ya', 0, mode='thm'), generate_path('ya', 1, mode='thm'),
    generate_path('zr', 0, mode='thm'), generate_path('zr', 1, mode='thm'),
    generate_path('zy', 0, mode='thm'), generate_path('zy', 1, mode='thm'),
]

# use data augmentation

isDataGen = True

for i, path in enumerate(specimen_list):
    # generate orignial data
    print('Processing specimen #' + str(i) + ' ' + str(path[0]) + '__________________________________')
    thm_preprocess(path)
    thm_preprocess(path, augmentation=['trans'])


# just plot the thing
# for i, path in enumerate(specimen_list):
#     # generate orignial data
#     print('Processing specimen #' + str(i) + '__________________________________')
#     idp_preprocess(path, is_plot=True)