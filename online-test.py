from __future__ import print_function

import numpy as np
from keras.callbacks import Callback
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy.random import choice



def prepare_sequences(x_train, y_train, window_length):
    windows = []
    windows_y = []
    for i, sequence in enumerate(x_train):
        len_seq = len(sequence)
        for window_start in range(0, len_seq - window_length + 1):
            window_end = window_start + window_length
            window = sequence[window_start:window_end]
            windows.append(window)
            windows_y.append(y_train[i])
    return np.array(windows), np.array(windows_y)

USE_SEQUENCES = False
USE_STATELESS_MODEL = False

# you can all the four possible combinations
# USE_SEQUENCES and USE_STATELESS_MODEL

max_len = 20
batch_size = 1

N_train = 1000
N_test = 200

X_train = np.zeros((N_train, max_len))
X_test = np.zeros((N_test, max_len))

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

y_train = np.zeros((N_train, 1))
y_test = np.zeros((N_test, 1))

one_indexes = choice(a=N_train, size=N_train / 2, replace=False)
X_train[one_indexes, 0] = 1
y_train[one_indexes] = 1

one_indexes = choice(a=N_test, size=N_test / 2, replace=False)
X_test[one_indexes, 0] = 1
y_test[one_indexes] = 1


class ResetStatesCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_batch_begin(self, batch, logs={}):
        if self.counter % max_len == 0:
            self.model.reset_states()
        self.counter += 1


if USE_SEQUENCES:
    max_len = 10
    X_train, y_train = prepare_sequences(X_train, y_train, window_length=max_len)
    X_test, y_test = prepare_sequences(X_test, y_test, window_length=max_len)

X_train = np.expand_dims(X_train, axis=2)  # input dim is 1. Timesteps is the sequence length.
X_test = np.expand_dims(X_test, axis=2)

print('sequences_x_train shape:', X_train.shape)
print('sequences_y_train shape:', y_train.shape)

print('sequences_x_test shape:', X_test.shape)
print('sequences_y_test shape:', y_test.shape)

if USE_STATELESS_MODEL:
    print('Build STATELESS model...')
    model = Sequential()
    model.add(LSTM(10, input_shape=(max_len, 1), return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
              validation_data=(X_test, y_test), shuffle=False, callbacks=[ResetStatesCallback()])

    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print('___________________________________')
    print('Test score:', score)
    print('Test accuracy:', acc)
else:
    # STATEFUL MODEL
    print('Build STATEFUL model...')
    model = Sequential()
    model.add(LSTM(10,
                   batch_input_shape=(1, 1, 1), return_sequences=False,
                   stateful=True))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    x = np.expand_dims(np.expand_dims(X_train.flatten(), axis=1), axis=1)
    y = np.expand_dims(np.array([[v] * max_len for v in y_train.flatten()]).flatten(), axis=1)
    # model.fit(x,
    #           y,
    #           callbacks=[ResetStatesCallback()],
    #           batch_size=1,
    #           shuffle=False)

    print('Train...')
    for epoch in range(15):
        mean_tr_acc = []
        mean_tr_loss = []
        for i in range(len(X_train)):
            y_true = y_train[i]
            for j in range(max_len):
                tr_loss, tr_acc = model.train_on_batch(np.expand_dims(np.expand_dims(X_train[i][j], axis=1), axis=1),
                                                       np.array([y_true]))
                mean_tr_acc.append(tr_acc)
                mean_tr_loss.append(tr_loss)
            model.reset_states()

        print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
        print('loss training = {}'.format(np.mean(mean_tr_loss)))
        print('___________________________________')

        mean_te_acc = []
        mean_te_loss = []
        for i in range(len(X_test)):
            for j in range(max_len):
                te_loss, te_acc = model.test_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=1), axis=1),
                                                      y_test[i])
                mean_te_acc.append(te_acc)
                mean_te_loss.append(te_loss)
            model.reset_states()

            for j in range(max_len):
                y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=1), axis=1))
            model.reset_states()

        print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
        print('loss testing = {}'.format(np.mean(mean_te_loss)))
        print('___________________________________')