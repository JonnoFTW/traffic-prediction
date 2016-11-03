from __future__ import print_function

from datetime import datetime, timedelta
import numpy as np
import pyprind

import theano
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


from metrics import MASE, mean_absolute_percentage_error, median_percentage_error, rmse, geh, mape
from utils import load_data, train_test_split, BestWeight, ResetStatesCallback


EPS = 1e-6

def step_data(FPATH, end_date=None, use_sensors=None):
    all_data = load_data(FPATH, EPS, use_sensors=use_sensors)
    return all_data


def chunks(x, y, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(x), n):
        yield x[i:i + n], y[i:i + n]


def do_model(all_data, steps):
    _steps = steps
    print("steps:", _steps)
    # all_data = all_data[:100]
    features = all_data[:-_steps]
    labels = all_data[_steps:, -1:]

    X_train = features
    Y_train = labels

    optimiser = 'adam'
    hidden_neurons = 332
    loss_function = 'mse'
    dropout = 0.0923
    inner_hidden_neurons = 269
    dropout_inner = 0.2269

    batch_size = 1

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    print("X train shape:\t", X_train.shape)
    # print("Y train shape:\t", Y_train.shape)
    # print("Y test shape:\t", Y_test.shape)
    # print("Steps:\t", _steps)
    in_neurons = X_train.shape[2]

    out_neurons = 1

    model = Sequential()
    if 'gpu' in theano.config.device:
        gpu_cpu = 'gpu'
    else:
        gpu_cpu = 'cpu'
    model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, batch_input_shape=(1,1, in_neurons) ,return_sequences=True, init='uniform',
                   consume_less=gpu_cpu, stateful=True))
    model.add(Dropout(dropout))

    dense_input = inner_hidden_neurons
    model.add(LSTM(output_dim=dense_input, input_dim=hidden_neurons, return_sequences=False, consume_less=gpu_cpu, stateful=True))
    model.add(Dropout(dropout_inner))
    model.add(Activation('relu'))

    model.add(Dense(output_dim=out_neurons, input_dim=dense_input))
    model.add(Activation('relu'))

    model.compile(loss=loss_function, optimizer=optimiser)
    # run through all data up to 23 April, 2013
    progress = pyprind.ProgBar(len(X_train), width=50, stream=1)
    mean_tr_loss = []
    for x_chunk, y_chunk in chunks(X_train, Y_train, batch_size):

        tr_loss = model.train_on_batch(x_chunk, y_chunk)
        mean_tr_loss.append(tr_loss)
        model.reset_states()
        progress.update()

    print("\nTraining Loss: {}".format(np.mean(mean_tr_loss)))
    return model


if __name__ == "__main__":
    start = datetime.now()
    file_path = '/scratch/Dropbox/PhD/htm_models_adelaide/engine/lane_data_3002_3001.csv'
    # print ("Examining", file_path)
    # data = step_data(file_path)#, end_date=datetime(2013, 4, 23), use_sensors=5)
    # fname = file_path.split('/')[-1]
    # print (fname)
    # model = do_model(data, 1)
    model = load_model('models/keras_1_step_3002_online_pre_test.h5')
    predict_data = load_data(file_path, EPS, use_datetime=True, load_from=datetime(2013, 4, 23), use_sensors=[5], end_date=datetime(2013, 6, 15))

    true_x = predict_data[:, 0]
    true_y = predict_data[:, 1].astype(np.float32)
    # replace 2046/2047 values with 50
    true_y[true_y > 2045] = -1
    pred_xy = []
    progress = pyprind.ProgBar(len(true_x[:-1]), width=50, stream=1)
    # flow_val = 8
    for idx, dt in enumerate(true_x[:-1]):
        in_row = [[
            dt.weekday(),
            # is weekend
            int(dt.weekday() in [5, 6]),
            dt.hour,
            dt.minute,
            max(1, true_y[idx])
        ]]
        npa = np.array([in_row])
        pred = model.predict(np.array([in_row]))
        model.reset_states()

        model.train_on_batch(npa, np.array([true_y[idx+1]]).reshape((1, 1)))
        model.reset_states()

        pred_xy.append((dt+timedelta(minutes=5), pred[0]))
        progress.update()

    pred_xy = np.array(pred_xy)
    pred_x = np.reshape(pred_xy[:,0], (-1,1))
    pred_y = np.reshape(pred_xy[:,1].astype(dtype=np.float32), (-1,1))
    true_y_max = np.copy(true_y)[:-1]
    true_y_max[true_y_max == 0] = 1
    print("PredY",pred_y.shape)
    print("TrueT_max", true_y_max.shape)
    print("GEH:  ", geh (true_y_max, pred_y))
    print("MAPE: ", mape(true_y_max, pred_y))
    print("RMSE: ", rmse(true_y_max, pred_y))

    import matplotlib.pyplot as plt
    plt.plot(true_x, true_y, 'b-', label='Readings')
    plt.plot(pred_x, pred_y, 'r-', label='LSTM-Online Predictions')
    df = "%A %d %B, %Y"
    plt.title("3002: Traffic Flow from {} to {}".format(true_x[0].strftime(df), true_x[-1].strftime(df)))
    plt.legend()

    plt.ylabel("Vehicles/ 5 min")
    plt.xlabel("Time")
    plt.show()
