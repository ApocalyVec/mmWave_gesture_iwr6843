import os
import pickle

import numpy as np
from utils.path_utils import idp_preprocess, generate_path, thm_preprocess

specimen_list = [
    generate_path('zl', 0, mode='thm'), generate_path('zl', 1, mode='thm'),
    generate_path('zl', 2, mode='thm'),
    generate_path('zy', 0, mode='thm'), generate_path('zy', 1, mode='thm'),
    generate_path('zy', 2, mode='thm'),
]

# use data augmentation

isDataGen = True

for i, path in enumerate(specimen_list):
    # generate orignial data
    print('Processing specimen #' + str(i) + ' ' + str(path[0]) + '__________________________________')
    thm_preprocess(path)

# just plot the thing
# for i, path in enumerate(specimen_list):
#     # generate orignial data
#     print('Processing specimen #' + str(i) + '__________________________________')
#     idp_preprocess(path, is_plot=True)