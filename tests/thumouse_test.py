from keras.engine.saving import load_model
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

from utils.path_utils import generate_train_val_ids
from sklearn.preprocessing import MinMaxScaler

regressor = load_model('D:/PycharmProjects/mmWave_gesture_iwr6843/models/thm_model.h5')

dataset_path = 'D:/alldataset/thm_dataset'
label_dict_path = 'D:/alldataset/thm_label_dict.p'

videoData_path = ''
video_frame_list = os.listdir(videoData_path)
video_frame_timestamps = list(map(lambda x: float(x.strip('.jpg')), video_frame_list))

label_dict = pickle.load(open(label_dict_path, 'rb'))

partition = generate_train_val_ids(0.1, dataset_path=dataset_path)

# get the Y and X
# take only the first 100 for plotting
X_test = []
Y_test = []

for i, val_sample in enumerate(partition['train']):
    print('Reading ' + str(i) + ' of 100')

    # finds the corresponding picture
    # this_timestamp =
    # closest_video_timestamp = min(video_frame_timestamps,
    #                               key=lambda x: abs(x - this_timestamp))
    # closest_video_path = os.path.join(videoData_path, str(closest_video_timestamp) + '.jpg')
    # closest_video_img = Image.open(closest_video_path)

    if i < 25:
        X_test.append(np.load(os.path.join(dataset_path, val_sample + '.npy')))
        Y_test.append(label_dict[val_sample])
    else:
        break
X_test = np.asarray(X_test)

# make the prediction
Y_predict = regressor.predict(X_test)

# inverse scale
mms = pickle.load(open('D:/PycharmProjects/mmWave_gesture_iwr6843/models/scalers/thm_scaler.p', 'rb'))
Y_predict = mms.inverse_transform(Y_predict)
Y_test = mms.inverse_transform(Y_test)

# plot the result
fig = plt.figure()
fig.set_size_inches(10, 5.5)
fig.suptitle('X Displacement')
plt.plot(Y_test[:, 0], label='CV Predicted X Displacement (Ground Truth)')
plt.plot(Y_predict[:, 0], label='X Displacement from Radar Tracking')
plt.legend()
plt.show()

fig = plt.figure()
fig.set_size_inches(15, 5.5)
fig.suptitle('Y Displacement')
plt.plot(Y_test[:, 1], label='CV Predicted Y Displacement (Ground Truth)')
plt.plot(Y_predict[:, 1], label='Y Displacement from Radar Tracking')
plt.legend()
plt.show()
