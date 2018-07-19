from __future__ import print_function

from collections import OrderedDict
from datetime import datetime
import numpy as np
import pluck as pluck
import tabulate

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional, quniform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from metrics import MASE, mean_absolute_percentage_error, median_percentage_error, rmse, smape, geh
from utils import load_data, train_test_split, check_gpu, BestWeight


def step_data():
    EPS = 1e-6
    fpath = get_fpath()
    print("Loading", fpath)
    all_data = load_data(fpath, EPS)
    return all_data


def fit_to_batch(arr, b_size):
    lim = len(arr) - (len(arr) % b_size)
    return arr[:lim]


def do_model(all_data):
    _steps, tts_factor, num_epochs = get_steps_extra()
    # features = all_data[:-_steps]
    # labels = all_data[_steps:, 4:]
    # tts = train_test_split(features, labels, test_size=0.4)
    # X_train = tts[0]
    # X_test = tts[1]
    # Y_train = tts[2].astype(np.float64)
    # Y_test = tts[3].astype(np.float64)
    split_pos = int(len(all_data) * tts_factor)
    train_data, test_data = all_data[:split_pos], all_data[split_pos:]
    dataX, dataY, fields = create_dataset(test_data, 1, _steps)

    optimiser = {{choice(['adam', 'rmsprop'])}}
    hidden_neurons = int({{quniform(16, 256, 4)}})
    loss_function = 'mse'
    batch_size = int({{quniform(1, 10, 1)}})
    dropout = {{uniform(0, 0.5)}}
    dropout_dense = {{uniform(0, 0.5)}}
    hidden_inner_factor = {{uniform(0.1, 1.9)}}
    inner_hidden_neurons = int(hidden_inner_factor * hidden_neurons)
    dropout_inner = {{uniform(0, 0.5)}}

    dataX = fit_to_batch(dataX, batch_size)
    dataY = fit_to_batch(dataY, batch_size)

    extra_layer = {{choice([True, False])}}
    if not extra_layer:
        dropout_inner = 0

    # X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    # X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # print("X train shape:\t", X_train.shape)
    # print("X test shape:\t", X_test.shape)
    # print("Y train shape:\t", Y_train.shape)
    # print("Y test shape:\t", Y_test.shape)
    print("Steps:\t", _steps)
    print("Extra layer:\t", extra_layer)
    print("Batch size:\t", batch_size)

    # in_neurons = X_train.shape[2]

    out_neurons = 1

    model = Sequential()
    best_weight = BestWeight()
    model.add(LSTM(
        units=hidden_neurons,
        batch_input_shape=(batch_size, 1, fields),
        return_sequences=extra_layer,
        stateful=True,
        dropout=dropout))
    model.add(Activation('relu'))

    if extra_layer:
        dense_input = inner_hidden_neurons
        model.add(LSTM(units=dense_input,
                       # input_shape=hidden_neurons,
                       stateful=True,
                       return_sequences=False,
                       dropout=dropout_inner))
        model.add(Activation('relu'))

    model.add(Dense(units=out_neurons, activation='relu'))
    model.add(Dropout(dropout_dense))
    model.compile(loss=loss_function, optimizer=optimiser)

    history = model.fit(
        dataX, dataY,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.3,
        shuffle=False,
        callbacks=[best_weight]
    )

    model.set_weights(best_weight.get_best())
    X_test, Y_test, _fields = create_dataset(test_data, 1, _steps)
    X_test, Y_test = fit_to_batch(X_test, batch_size), fit_to_batch(Y_test, batch_size)
    predicted = model.predict(X_test, batch_size=batch_size) + EPS
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
        ('extra_layer', extra_layer),
        ('extra_layer_dropout', dropout_inner),
        ('dropout_dense', dropout_dense),
        ('extra_layer_neurons', inner_hidden_neurons),
        ('loss function', loss_function)
        # 'history': history.history
    ])
    print(metrics)
    return {'loss': -rmse_val, 'status': STATUS_OK, 'metrics': metrics}


def get_steps_extra():
    return int(sys.argv[1]), float(sys.argv[3]), int(sys.argv[5])


def get_fpath():
    return sys.argv[2]


def create_dataset(dataset, lookback=1, steps=1):
    row_count, fields = dataset.shape

    dataX = np.empty((row_count - lookback - 1, lookback, fields), dtype=np.double)
    dataY = np.empty((row_count - lookback - 1, 1), dtype=np.double)
    for i in range(row_count - lookback - 1):
        dataX[i] = dataset[i:(i + lookback), :]
        dataY[i] = dataset[i + lookback, 0]
    return dataX, dataY, fields


if __name__ == "__main__":
    import pymongo
    import sys, os

    try:
        steps = int(sys.argv[1])
        file_path = sys.argv[2]
        tts = float(sys.argv[3])
        train_epochs = int(sys.argv[4])
        model_trials = int(sys.argv[5])
    except (IndexError, ValueError) as e:
        exit("Usage is: main.py <steps eg. 1> <file_path> <train_test_split eg. 0.75> <num_epochs eg. 10> <model_trials eg. 20>")
    mongo_str = os.getenv('pymongo_conn', None)
    if not mongo_str:
        exit("Please Provide `pymongo_conn` environment variable")
    client = pymongo.MongoClient(mongo_str)
    print("Started: " + str(datetime.now()))
    trials = Trials()
    print("optimising network for {} steps".format(steps))

    best_run, best_model = optim.minimize(
        model=do_model,
        data=step_data,
        algo=tpe.suggest,
        max_evals=model_trials,
        trials=trials,
        functions=[get_steps_extra, get_fpath, create_dataset, fit_to_batch]
    )
    # put the trial results in
    trial_results = pluck.pluck(trials.results, 'metrics')
    results = client['mack0242']['hyperopt']
    results.insert_many(trial_results)

    # print (best_run, best_model, trials.trials)
    print(tabulate.tabulate(sorted(trial_results, key=lambda x: (x['steps'], x['rmse'])), headers='keys'))
    print("Finished: " + str(datetime.now()))
