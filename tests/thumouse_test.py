from keras.engine.saving import load_model
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

from utils.path_utils import generate_train_val_ids
from sklearn.preprocessing import MinMaxScaler

regressor = load_model('D:/thumouse/trained_models/bestSoFar_thuMouse_CRNN2019-08-19_22-29-26.855251.h5')

dataset_path = 'D:/thumouse/dataset'
label_dict_path = 'D:/thumouse/labels/label_dict.p'

label_dict = pickle.load(open(label_dict_path, 'rb'))

partition = generate_train_val_ids(0.1, dataset_path=dataset_path)

# get the Y and X
# take only the first 100 for plotting
X_test = []
Y_test = []

for i, val_sample in enumerate(partition['validation']):
    print('Reading ' + str(i) + ' of 100')
    if i < 100:
        X_test.append(np.load(os.path.join(dataset_path, val_sample + '.npy')))
        Y_test.append(label_dict[val_sample])
    else:
        break
X_test = np.asarray(X_test)

# make the prediction
Y_predict = regressor.predict(X_test)

# inverse scale
mms = pickle.load(open('F:/thumouse/scaler/thm_scaler.p', 'rb'))
Y_predict = mms.inverse_transform(Y_predict)
Y_test = mms.inverse_transform(Y_test)

# plot the result
plt.plot(Y_test[:, 0])
plt.plot(Y_predict[:, 0])
plt.show()

plt.plot(Y_test[:, 1])
plt.plot(Y_predict[:, 1])
plt.show()
