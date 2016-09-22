from __future__ import print_function

from collections import OrderedDict
from datetime import datetime
import numpy as np
import pluck as pluck
import tabulate

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from metrics import MASE, mean_absolute_percentage_error, median_percentage_error, rmse, smape, geh
from utils import load_data, train_test_split


def step_data():
    EPS = 1e-6
    all_data = load_data('/scratch/Dropbox/PhD/htm_models_adelaide/engine/lane_data.csv', EPS)
    return all_data


def do_model(all_data):
    _steps = steps
    print("steps:", _steps)
    features = all_data[:-_steps]
    labels = all_data[_steps:, 4:]
    tts = train_test_split(features, labels, test_size=0.4)
    X_train = tts[0]
    X_test = tts[1]
    Y_train = tts[2]
    Y_test = tts[3]
    optimiser = 'adam'
    hidden_neurons = {{choice([212, 230, 256])}}
    loss_function = 'mse'
    batch_size = {{choice([128, 148, 156, 164, 196])}}
    dropout = {{uniform(0, 1)}}

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    Y_train = Y_train.astype(np.float64)
    Y_test = Y_test.astype(np.float64)
    print("X train shape:\t", X_train.shape)
    print("X test shape:\t", X_test.shape)
    print("Y train shape:\t", Y_train.shape)
    print("Y test shape:\t", Y_test.shape)
    print("Steps:\t", _steps)
    in_neurons = X_train.shape[2]

    out_neurons = 1

    class BestWeight(Callback):
        def __init__(self, monitor='val_loss', mode='auto', verbose=0):
            super(BestWeight, self).__init__()
            self.monitor = monitor
            self.mode = mode
            self.best_weights = None
            self.verbose = verbose
            if mode == 'min':
                self.monitor_op = np.less
                self.best = np.Inf
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                if 'acc' in self.monitor:
                    self.monitor_op = np.greater
                    self.best = -np.Inf
                else:
                    self.monitor_op = np.less
                    self.best = np.Inf

        def get_best(self):
            return self.best_weights

        def on_epoch_end(self, epoch, logs={}):
            current = logs.get(self.monitor)
            if current is not None and self.monitor_op(current, self.best):
                self.best_weights = self.model.get_weights()
                if self.verbose > 0:
                    print("Epoch {}: {} improved from {} to {}".format(epoch, self.monitor, self.best, current))
                self.best = current

    model = Sequential()
    best_weight = BestWeight()
    model.add(LSTM(output_dim=hidden_neurons, input_dim=X_test.shape[2], return_sequences=False, init='uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim=out_neurons, input_dim=hidden_neurons, ))
    model.add(Activation('relu'))
    model.compile(loss=loss_function, optimizer=optimiser)

    history = model.fit(
        X_train, Y_train,
        verbose=0,
        batch_size=batch_size,
        nb_epoch=2,
        validation_split=0.3,
        shuffle=False,
        callbacks=[best_weight]
    )

    model.set_weights(best_weight.get_best())

    predicted = model.predict(X_test) + EPS
    rmse_val = rmse(Y_test, predicted)
    metrics = OrderedDict([
        ('hidden', hidden_neurons),
        ('steps', _steps),
        ('geh', geh(Y_test, predicted)),
        ('rmse', rmse_val),
        ('mape', mean_absolute_percentage_error(Y_test, predicted)),
        # ('smape', smape(predicted, _Y_test)),
        ('median_pe', median_percentage_error(predicted, Y_test)),
        # ('mase', MASE(_Y_train, _Y_test, predicted)),
        ('mae', mean_absolute_error(y_true=Y_test, y_pred=predicted)),
        ('batch_size', batch_size),
        ('optimiser', optimiser),
        ('dropout', dropout),
        ('loss function', loss_function)
        # 'history': history.history
    ])
    # print(metrics)
    return {'loss': rmse_val, 'status': STATUS_OK, 'metrics': metrics}


if __name__ == "__main__":
    results = []
    for i in [1, 3, 6, 9, 12]:
        trials = Trials()
        best_run, best_model = optim.minimize(
            model=do_model,
            data=step_data,
            algo=tpe.suggest,
            max_evals=1,
            trials=trials,
            extra={'steps':i}
        )
        results.extend(pluck.pluck(trials.results, 'metrics'))

    # print (best_run, best_model, trials.trials)
    print(tabulate.tabulate(sorted(results, key=lambda x: (x['steps'], x['rmse'])), headers='keys'))