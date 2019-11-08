#%%

import pickle
import os

import numpy as np
from keras.models import load_model
from numpy import ma
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from utils.data_utils import plot_confusion_matrix
from utils.path_utils import generate_train_val_ids

idp_model_path = 'D:/trained_models/bestSoFar_indexPen_CRNN2019-11-01_19-29-25.370584.h5'
idp_model = load_model(idp_model_path)


label_dict_path = 'D:/alldataset/idp_label_dict.p'
dataset_path = 'D:/alldataset/idp_dataset'

labels = pickle.load(open(label_dict_path, 'rb'))

classifying_labels = list(range(10))

X = []
Y = []
# for i, data in enumerate(sorted(os.listdir(dataset_path), key=lambda x: int(x.strip('.npy').split('_')[2]))):
for i, data in enumerate(os.listdir(dataset_path)):
    lb = labels[os.path.splitext(data)[0]]
    # if lb in classifying_labels:  # this is not an 'O'
    print('Loading ' + str(i) + ' of ' + str(len(os.listdir(dataset_path))))
    X.append(np.load(os.path.join(dataset_path, data)))
    Y.append(labels[os.path.splitext(data)[0]])
X = np.asarray(X)
Y = np.asarray(Y)

encoder = OneHotEncoder(categories='auto')
Y = encoder.fit_transform(np.expand_dims(Y, axis=1)).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=3, shuffle=True)

y_pred = idp_model.predict(np.asarray(X_train), batch_size=8)

plot_confusion_matrix(Y_train.argmax(axis=1), y_pred.argmax(axis=1), classes=np.asarray(classifying_labels), title='IndexPen Confusion Matrix')
plt.show()

# correct_mask = (y_pred != y_test)
# incorrect_mask = (y_pred == y_test)
#
# y_correctly_masked_nn_output = ma.array(np.amax(y_pred_nn_output, axis=1), mask=correct_mask)
# y_correct_nn_output = y_correctly_masked_nn_output.compressed()
#
# y_wrongfully_masked_nn_output = ma.array(np.amax(y_pred_nn_output, axis=1), mask=incorrect_mask)
# y_wrong_nn_output = y_wrongfully_masked_nn_output.compressed()
#
# sorted_correct = np.sort(y_correct_nn_output)
# sorted_wrong = np.sort(y_wrong_nn_output)
#
# wrong_mean = np.mean(y_wrong_nn_output)
# correct_mean = np.mean(y_correct_nn_output)
#
# plt.plot(sorted_wrong)
# plt.plot(sorted_correct)
#
# plt.show()

# Plot ROC --------------------------------------------------
# from sklearn.metrics import roc_curve
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
