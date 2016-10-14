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
from hyperas.distributions import choice, uniform, conditional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from metrics import MASE, mean_absolute_percentage_error, median_percentage_error, rmse, smape, geh
from utils import load_data, train_test_split, check_gpu, BestWeight


def step_data():
    EPS = 1e-6
    all_data = load_data(FPATH, EPS)
    return all_data




def do_model(all_data):
    _steps = steps
    print("steps:", _steps)
    features = all_data[:-_steps]
    labels = all_data[_steps:, 4:]
    tts = train_test_split(features, labels, test_size=0.4)
    X_train = tts[0]
    X_test = tts[1]
    Y_train = tts[2].astype(np.float64)
    Y_test = tts[3].astype(np.float64)
    optimiser = 'adam'
    hidden_neurons = {{choice([256, 300, 332])}} #tested already on : 128, 196, 212, 230, 244,
    loss_function = 'mse'
    batch_size = {{choice([96, 105, 128])}} # already did 148, 156, 164, 196
    dropout = {{uniform(0, 0.1)}}
    hidden_inner_factor = {{uniform(0.1, 1.1)}}
    inner_hidden_neurons = int(hidden_inner_factor * hidden_neurons)
    dropout_inner = {{uniform(0,1)}}

    extra_layer = {{choice([True, False])}}
    if not extra_layer:
        dropout_inner = 0

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print("X train shape:\t", X_train.shape)
    # print("X test shape:\t", X_test.shape)
    # print("Y train shape:\t", Y_train.shape)
    # print("Y test shape:\t", Y_test.shape)
    # print("Steps:\t", _steps)
    print("Extra layer:\t", extra_layer)
    in_neurons = X_train.shape[2]

    out_neurons = 1


    model = Sequential()
    gpu_cpu = 'gpu'
    best_weight = BestWeight()
    dense_input = hidden_neurons
    model.add(LSTM(output_dim=hidden_neurons, input_dim=X_test.shape[2], return_sequences=extra_layer, init='uniform',
                   consume_less=gpu_cpu))
    model.add(Dropout(dropout))

    if extra_layer:
        dense_input = inner_hidden_neurons
        model.add(LSTM(output_dim=dense_input, input_dim=hidden_neurons, return_sequences=False, consume_less=gpu_cpu))
        model.add(Dropout(dropout_inner))
        model.add(Activation('relu'))

    model.add(Dense(output_dim=out_neurons, input_dim=dense_input))
    model.add(Activation('relu'))
    model.compile(loss=loss_function, optimizer=optimiser)

    history = model.fit(
        X_train, Y_train,
        verbose=0,
        batch_size=batch_size,
        nb_epoch=30,
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
        ('extra_layer', extra_layer),
        ('extra_layer_dropout', dropout_inner),
        ('extra_layer_neurons', inner_hidden_neurons),
        ('loss function', loss_function)
        # 'history': history.history
    ])
    # print(metrics)
    return {'loss': -rmse_val, 'status': STATUS_OK, 'metrics': metrics}


if __name__ == "__main__":
    import pymongo
    import sys, os
    try:
        steps = int(sys.argv[1])
        file_path = sys.argv[2]
    except IndexError:
        quit("Usage is: main.py <steps> <file_path>")
    mongo_str = os.getenv('pymongo_conn', None)
    if not mongo_str:
        quit("Please Provide `pymongo_conn` environment variable")
    client = pymongo.MongoClient(mongo_str)
    print("Started: "+str(datetime.now()))
    trials = Trials()
    print("optimising network for {} steps".format(steps))
    best_run, best_model = optim.minimize(
        model=do_model,
        data=step_data,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials,
        extra={'steps': steps, 'FPATH': file_path}
    )
    # put the trial results in
    trial_results = pluck.pluck(trials.results, 'metrics')
    results = client['mack0242']['hyperopt']
    results.insert_many(trial_results)

    # print (best_run, best_model, trials.trials)
    print(tabulate.tabulate(sorted(trial_results, key=lambda x: (x['steps'], x['rmse'])), headers='keys'))
    print("Finished: "+str(datetime.now()))
