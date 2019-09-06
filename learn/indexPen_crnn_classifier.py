import datetime
import pickle

from keras import Sequential, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv3D, MaxPooling3D, Flatten, TimeDistributed, LSTM, Dropout, Dense, BatchNormalization, \
    LeakyReLU
from keras.layers import Conv3D, MaxPooling3D, Flatten, TimeDistributed, LSTM, Dropout, Dense, BatchNormalization, \
    LeakyReLU
from keras.regularizers import l2
from keras.engine.saving import load_model

import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from learn.classes import indexPenDataGen
from utils.path_utils import generate_train_val_ids

pre_trained_path = 'D:/code/DoubleMU/models/palmPad_model.h5'
epochs = 5000
is_use_pre_train = False
num_classes = 5
timesteps = 80

if __name__ == '__main__':
    label_dict_path = 'D:/alldataset/idp_label_dict.p'
    dataset_path = 'D:/alldataset/idp_dataset'
    labels = pickle.load(open(label_dict_path, 'rb'))

    ## Generators
    X = []
    Y = []
    # for i, data in enumerate(sorted(os.listdir(dataset_path), key=lambda x: int(x.strip('.npy').split('_')[2]))):
    for i, data in enumerate(os.listdir(dataset_path)):
        X.append(np.load(os.path.join(dataset_path, data)))
        Y.append(labels[os.path.splitext(data)[0]])
    X = np.asarray(X)
    Y = np.asarray(Y)

    encoder = OneHotEncoder(categories='auto')
    Y = encoder.fit_transform(np.expand_dims(Y, axis=1)).toarray()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=None, shuffle=False)

    # Build the RNN ###############################################
    if not is_use_pre_train:
        classifier = Sequential()
        classifier.add(
            TimeDistributed(
                Conv3D(filters=16, kernel_size=(3, 3, 3), data_format='channels_first', input_shape=(1, 25, 25, 25),
                       kernel_regularizer=l2(0.0005), kernel_initializer='random_uniform'), input_shape=(timesteps, 1, 25, 25, 25)))
        # classifier.add(TimeDistributed(LeakyReLU(alpha=0.1)))
        classifier.add(TimeDistributed(BatchNormalization()))
        classifier.add(TimeDistributed(Conv3D(filters=16, kernel_size=(3, 3, 3), data_format='channels_first', kernel_regularizer=l2(0.0005))))
        # classifier.add(TimeDistributed(LeakyReLU(alpha=0.1)))
        classifier.add(TimeDistributed(BatchNormalization()))

        classifier.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2))))

        # classifier.add(TimeDistributed(Conv3D(filters=128, kernel_size=(3, 3, 3), data_format='channels_first')))
        # classifier.add(TimeDistributed(LeakyReLU(alpha=0.1)))
        # classifier.add(TimeDistributed(Conv3D(filters=64, kernel_size=(1, 1, 1), data_format='channels_first')))
        # classifier.add(TimeDistributed(LeakyReLU(alpha=0.1)))
        # classifier.add(TimeDistributed(Conv3D(filters=128, kernel_size=(3, 3, 3), data_format='channels_first')))
        # classifier.add(TimeDistributed(LeakyReLU(alpha=0.1)))
        # classifier.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2))))

        classifier.add(TimeDistributed(Flatten()))

        classifier.add(LSTM(units=64, return_sequences=True, kernel_initializer='random_uniform'))
        classifier.add(Dropout(rate=0.5))

        classifier.add(LSTM(units=64, return_sequences=False, kernel_initializer='random_uniform'))
        classifier.add(Dropout(rate=0.5))

        # classifier.add(Dense(units=512))
        # classifier.add(LeakyReLU(alpha=0.1))
        # classifier.add(Dropout(rate=0.2))

        # classifier.add(Dense(units=512))
        # classifier.add(LeakyReLU(alpha=0.1))
        classifier.add(Dense(units=128, activation='sigmoid'))

        classifier.add(Dense(num_classes, activation='softmax', kernel_initializer='random_uniform'))

        # adam = optimizers.adam(lr=1e-6, decay=1e-2 / epochs)
        sgd = optimizers.SGD(lr=5e-5, momentum=0.9, decay=1e-2 / epochs, nesterov=True, clipvalue=0.5)

        classifier.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print('Using Pre-trained Model: ' + pre_trained_path)
        classifier = load_model(pre_trained_path)

    # add early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=175)
    mc = ModelCheckpoint(
        'D:/trained_models/bestSoFar_indexPen_CRNN' + str(datetime.datetime.now()).replace(':', '-').replace(' ',
                                                                                                             '_') + '.h5',
        monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    history = classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5000,
                             batch_size=8, callbacks=[es, mc], verbose=1,)

    import matplotlib.pyplot as plt

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