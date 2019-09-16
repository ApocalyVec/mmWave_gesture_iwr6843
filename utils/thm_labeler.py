import pickle
import os

from utils.data_utils import merge_dict

pred_dir = 'C:/Users/S-Vec/Downloads/R_predict_new/R_predict'
dataset_dir = 'D:/alldataset/thm_dataset'

pred_file_list = os.listdir(pred_dir)

pred_dicts = [pickle.load(open(os.path.join(pred_dir, x), 'rb')) for x in pred_file_list]
merged_pred_dict = merge_dict(pred_dicts)

# check if any label is missing
dataset = [x.strip('.npy') for x in os.listdir(dataset_dir)]

data_without_label = [x for x in dataset if (x.split('_')[0] + '_' + x.split('_')[1]) not in merged_pred_dict.keys()]

result_dict = dict()

for key, value in merged_pred_dict.items():

    result_dict[key] = value[3:]
    result_dict[key + '_trans'] = value[3:]

pickle.dump(data_without_label, open('D:/alldataset/thm_data_without_label.p', 'wb'))
pickle.dump(result_dict, open('D:/alldataset/thm_label_dict.p', 'wb'))