#%%

import pickle
import os

import numpy as np
from keras.models import load_model
from numpy import ma
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from utils.data_utils import plot_confusion_matrix
from utils.path_utils import generate_train_val_ids

idp_model_path = 'D:\PycharmProjects\mmWave_gesture_iwr6843\models\palmPad_model.h5'
idp_model = load_model(idp_model_path)

label_dict = pickle.load(open('E:\indexPen\labels_old/label_dict.p', 'rb'))

dataset_path = 'E:\indexPen\dataset_old'
partition = generate_train_val_ids(0.1, dataset_path=dataset_path)

y_test = []
y_pred_nn_output = []

sample = None

for i, sample_path in enumerate(os.listdir(dataset_path)):
    print('Processing ' + str(i) + ' of ' + str(len(os.listdir(dataset_path))))

    if sample is not None:
        a = np.all(sample == np.load(os.path.join(dataset_path, sample_path)))
        assert not a

    sample = (np.load(os.path.join(dataset_path, sample_path)))

    y_pred_nn_output.append(idp_model.predict(np.expand_dims(sample, axis=0))[0])

    y_test.append(label_dict[os.path.splitext(sample_path)[0]])

y_test = np.asarray(y_test)
y_pred = np.argmax(y_pred_nn_output, axis=1)

plot_confusion_matrix(y_test, y_pred, classes=np.asarray(['A', 'D', 'L', 'M', 'P']), title='IndexPen Confusion Matrix')
plt.show()

correct_mask = (y_pred != y_test)
incorrect_mask = (y_pred == y_test)

y_correctly_masked_nn_output = ma.array(np.amax(y_pred_nn_output, axis=1), mask=correct_mask)
y_correct_nn_output = y_correctly_masked_nn_output.compressed()

y_wrongfully_masked_nn_output = ma.array(np.amax(y_pred_nn_output, axis=1), mask=incorrect_mask)
y_wrong_nn_output = y_wrongfully_masked_nn_output.compressed()

sorted_correct = np.sort(y_correct_nn_output)
sorted_wrong = np.sort(y_wrong_nn_output)

wrong_mean = np.mean(y_wrong_nn_output)
correct_mean = np.mean(y_correct_nn_output)

plt.plot(sorted_wrong)
plt.plot(sorted_correct)

plt.show()

# Plot ROC --------------------------------------------------
# from sklearn.metrics import roc_curve
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
