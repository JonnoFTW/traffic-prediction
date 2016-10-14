from __future__ import print_function

from collections import OrderedDict
from datetime import datetime, date
import numpy as np
import pluck as pluck
import tabulate

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from metrics import MASE, mean_absolute_percentage_error, median_percentage_error, rmse, geh
from utils import load_data, train_test_split, BestWeight


EPS = 1e-6
def step_data(FPATH):
    all_data = load_data(FPATH, EPS, limit=149976, use_sensors=[5])
    return all_data


def do_model(all_data, steps):
    _steps = steps
    print("steps:", _steps)
    features = all_data[:-_steps]
    labels = all_data[_steps:, -1:]
    tts = train_test_split(features, labels, test_size=0.4)
    X_train = tts[0]
    X_test = tts[1]
    Y_train = tts[2].astype(np.float64)
    Y_test = tts[3].astype(np.float64)
    optimiser = 'adam'
    hidden_neurons = 300
    loss_function = 'mse'
    batch_size = 105
    dropout = 0.056
    inner_hidden_neurons = 269
    dropout_inner = 0.22

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print("X train shape:\t", X_train.shape)
    print("X test shape:\t", X_test.shape)
    # print("Y train shape:\t", Y_train.shape)
    # print("Y test shape:\t", Y_test.shape)
    # print("Steps:\t", _steps)
    in_neurons = X_train.shape[2]

    out_neurons = 1



    model = Sequential()
    gpu_cpu = 'gpu'
    best_weight = BestWeight()
    model.add(LSTM(output_dim=hidden_neurons, input_dim=X_test.shape[2], return_sequences=True, init='uniform',
                   consume_less=gpu_cpu))
    model.add(Dropout(dropout))

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
        # ('hidden', hidden_neurons),
        ('steps', _steps),
        ('geh', geh(Y_test, predicted)),
        ('rmse', rmse_val),
        ('mape', mean_absolute_percentage_error(Y_test, predicted)),
        # ('smape', smape(predicted, _Y_test)),
        # ('median_pe', median_percentage_error(predicted, Y_test)),
        # ('mase', MASE(_Y_train, _Y_test, predicted)),
        # ('mae', mean_absolute_error(y_true=Y_test, y_pred=predicted)),
        # ('batch_size', batch_size),
        # ('optimiser', optimiser),
        # ('dropout', dropout),
        # ('extra_layer_dropout', dropout_inner),
        # ('extra_layer_neurons', inner_hidden_neurons),
        # ('loss function', loss_function)
        # 'history': history.history
    ])

    return metrics, model


if __name__ == "__main__":
    import sys, os
    pass
    try:
        file_path = sys.argv[1]
    except IndexError:
        quit("Usage is: evaluate.py <file_path_1> <file_path_2> ...")
    start = datetime.now()
    for file_path in sys.argv[1:]:
        print ("Examining", file_path)
        data = step_data(file_path)
        metrics = []
        fname = file_path.split('/')[-1]
        for i in [1, 3, 6, 9, 12]:
            metric_out, model = do_model(data, i)
            metrics.append(metric_out)
            model.save('models/keras_{}_step_{}.h5'.format(i, fname))
        # model has:       1  1.45893  14.3746  34.0476
        # print("Loading model")
        # model = load_model('best_sensor_5_with_calendar.h5')
        #
        print("Finished in "+str(datetime.now() - start))
        print(tabulate.tabulate([metrics], headers='keys', tablefmt="latex"))


        # print("Loading impute data")
        # predict_data = load_data(file_path, EPS, use_datetime=True, load_from=datetime(2013, 4, 23), use_sensors=[5])
        # true_x = predict_data[:(-288*26), 0]
        # true_y = predict_data[:(-288*26), 1]
        # # replace 2046/2047 values with 50
        # true_y[true_y > 2045] = -1
        # pred_y = []
        # # flow_val = 8
        # for idx, dt in enumerate(true_x):
        #     try:
        #         flow_val = true_y[idx+1]
        #     except IndexError:
        #         break
        #     pred = model.predict(np.array([[[
        #         dt.weekday(),
        #         # is weekend
        #         int(dt.weekday() in [5, 6]),
        #         # hour of day
        #         dt.isocalendar()[1],
        #         dt.hour,
        #         dt.minute,
        #         flow_val
        #     ]]]))
        #     # flow_val = pred[0][0]
        #     pred_y.append(pred[0][0])
        #
        #
        #
        # import matplotlib.pyplot as plt
        # plt.plot(true_x[:-1], true_y[:-1], 'b-', label='Readings')
        # plt.plot(true_x[:-1], pred_y, 'r-', label='Predictions')
        # df = "%A %d %B, %Y"
        # plt.title("3002: Traffic Flow from {} to {}".format(true_x[0].strftime(df), true_x[-1].strftime(df)))
        # plt.legend()
        #
        #
        # plt.ylabel("Vehicles/ 5 min")
        # plt.xlabel("Time")
        # plt.show()
