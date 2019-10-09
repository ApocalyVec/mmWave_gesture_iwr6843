import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Flatten, LSTM

from keras.layers import Dropout
from keras.optimizers import SGD
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics

import matplotlib.pyplot as plt
# smote randomly picks a point from the minority class and compute the k-nearest neighbors for
# this point. The synthetic points are added between the chosen point and its
# neighbors
from sklearn.utils import compute_class_weight, compute_sample_weight
import pydot


dataset_0 = np.load('0.npy')
dataset_1 = np.load('1.npy')

X = []
Y = []

for d0, d1 in zip(dataset_0, dataset_1):
    X.append(d0)
    Y.append(0)

    X.append(d1)
    Y.append(1)

X = np.asarray(X)
Y = np.asarray(Y)

X = (X - np.min(X))/(np.max(X) - np.min(X))

# separate training and test set
train_test_ratio = 0.2
X_train, X_test = X[int(len(X) * train_test_ratio):], X[:int(len(X) * train_test_ratio)]
Y_train, Y_test = Y[int(len(Y) * train_test_ratio):], Y[:int(len(Y) * train_test_ratio)]

# build and fit the network
classifier = Sequential()

classifier.add(Conv1D(filters=8, kernel_size=(3), data_format='channels_first', input_shape=(2, X_train.shape[2]), activation='relu', kernel_regularizer=l2(0.0005)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling1D(pool_size=2, data_format='channels_first'))

classifier.add(Conv1D(filters=8, kernel_size=(3), data_format='channels_first', activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling1D(pool_size=2, data_format='channels_first'))

classifier.add(Flatten())

classifier.add((Dense(units=16, activation='relu')))
classifier.add(Dropout(rate=0.2))

classifier.add((Dense(units=1, activation='sigmoid')))

epochs = 1000
adam = optimizers.adam(lr=1e-4, clipnorm=1., decay=1e-2/epochs)  # use half the learning rate as adam optimizer default
classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
history = classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), shuffle=True, epochs=epochs)

# plot train history

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = classifier.predict(X)

# Making the Confusion Matrix
matrix = metrics.confusion_matrix(Y.argmax(axis=1), y_pred.argmax(axis=1))

