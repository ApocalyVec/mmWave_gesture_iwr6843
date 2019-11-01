import os
import pickle

import numpy as np
from utils.path_utils import idp_preprocess, generate_path

specimen_list = []

subject_name_list = ['ya', 'zy', 'zl', 'am', 'sd']
num_sessions = 6

for sn in subject_name_list:
    [specimen_list.append(generate_path(subject_name=sn, case_index=i, mode='idp')) for i in range(num_sessions)]


# use data augmentation

isDataGen = True

for i, path in enumerate(specimen_list):
    # generate orignial data
    print('Processing specimen #' + str(i) + ' ' + str(path[0]) + '__________________________________')
    idp_preprocess(path, is_plot=False)
    # idp_preprocess(path, augmentation=['trans'])

    # idp_preprocess(path, augmentation=['clp', 'trans'])
    # idp_preprocess(path, augmentation=['rot'])
    # idp_preprocess(path,  augmentation=['scale'])
    #
    # idp_preprocess(path, augmentation=['trans', 'rot'])
    # idp_preprocess(path, augmentation=['trans', 'scale'])
    # idp_preprocess(path,  augmentation=['rot', 'scale'])
    #
    # idp_preprocess(path,  augmentation=['trans', 'rot', 'scale'])

# just plot the thing
# for i, path in enumerate(specimen_list):
#     # generate orignial data
#     print('Processing specimen #' + str(i) + '__________________________________')
#     idp_preprocess(path, is_plot=True)