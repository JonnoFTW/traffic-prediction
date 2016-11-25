from __future__ import print_function

from datetime import datetime, timedelta
import numpy as np
import pyprind

import theano
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


from metrics import MASE, mean_absolute_percentage_error, median_percentage_error, rmse, geh, mape
from utils import load_data, train_test_split, BestWeight, ResetStatesCallback, show_plot


EPS = 1e-6
sequence_length = 50

def step_data(FPATH, end_date=None, use_sensors=None, use_datetime=False):
    all_data = load_data(FPATH, EPS, use_sensors=use_sensors, use_datetime=use_datetime)
    return all_data


def chunks(x, y, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(x), n):
        yield x[i:i + n], y[i:i + n]


def do_model(all_data, steps, dates):
    _steps = steps
    # trim = 100
    # all_data = all_data[:trim]
    # dates = dates[:trim]
    print("steps:", _steps)
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
    model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, batch_input_shape=(1,1, in_neurons), return_sequences=True, init='uniform',
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
    seq = 0
    inputs = zip(X_train, Y_train)
    for idx, tup in enumerate(inputs):
        x_chunk, y_chunk = tup
        tr_loss = model.train_on_batch(np.array([x_chunk]), y_chunk)
        mean_tr_loss.append(tr_loss)
        seq += 1
        if seq % sequence_length == 0:
            model.reset_states()
        progress.update()
    # for x_chunk, y_chunk in chunks(X_train, Y_train, batch_size):
    # # for
    #     # we need to reset states when we have an error value
    #     if last_x is None:
    #
    #         last_x = x_chunk
    #
    #     tr_loss = model.train_on_batch(x_chunk, y_chunk)
    #
    #     mean_tr_loss.append(tr_loss)
    #     seq += 1
    #     if seq % sequence_length == 0:
    #         model.reset_states()
    #     progress.update()


    print("\nTraining Loss: {}".format(np.mean(mean_tr_loss)))
    return model

sensors = [5,6,7]
if __name__ == "__main__":
    start = datetime.now()
    import sys
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
    else:
        file_path = '/scratch/Dropbox/PhD/htm_models_adelaide/engine/lane_data_3002_3001.csv'
    print ("Examining", file_path)
    date_split = datetime(2013, 4, 23)
    data, dates = step_data(file_path, end_date=date_split, use_sensors=sensors, use_datetime=True)
    fname = file_path.split('/')[-1]
    print (fname)
    steps = 1
    model_name = 'models/keras_1_step_3002_online_no_state_reset.h5'
    model = do_model(data, steps, dates)
    # model.save(model_name)
    # model = load_model(model_name)
    true_x_row, true_x = load_data(file_path, EPS, use_datetime=True, load_from=date_split, use_sensors=sensors, end_date=datetime(2013, 6, 15), skip_error=False)
    # replace 2046/2047 values with 50
    pred_xy = []
    true_xy = []
    progress = pyprind.ProgBar(len(true_x[:-1]), width=50, stream=1)
    # flow_val = 8
    seq = 0
    inputs = zip(true_x_row, true_x)
    for idx, xy in enumerate(inputs[:-1]):
        row, dt = xy
        dt_next = dt+timedelta(minutes=5*steps)
        npa = np.array([[row]])
        if row[-1] > 300: # error value,
            true_xy.append((dt, np.nan))
            pred_xy.append((dt_next, np.nan))
        else:
            pred = model.predict(npa)
            seq += 1
            if seq % sequence_length == 0:
                model.reset_states()
            true_xy.append((dt, np.float32(row[-1])))
            model.train_on_batch(npa, np.array([true_x_row[idx+1][0]]).reshape((1, 1)))

            pred_xy.append((dt_next, pred[0]))
        progress.update()

    pred_xy = np.array(pred_xy)
    true_xy = np.array(true_xy)
    true_x = true_xy[:, 0]
    pred_x = np.reshape(pred_xy[:, 0], (-1, 1))
    pred_y = np.reshape(pred_xy[:, 1].astype(dtype=np.float32), (-1, 1))
    true_y = true_xy[:,1].astype(np.float32)
    true_y_max = np.copy(true_y)[:-1]
    true_y_max[true_y_max == 0] = 1
    np.savez('pred_data/3002-no-reset-on-error-all-sensor', true_x=true_x, true_y=true_y, pred_x=pred_x, pred_y=pred_y)
    true_y_max = true_y_max.reshape((true_y_max.shape[0],1))
    # print ("true_y_max", true_y_max.shape)
    # print("pred_y", pred_y.shape)
    print("GEH:  ", geh(true_y_max, pred_y[:-1]))
    print("MAPE: ", mape(true_y_max, pred_y[:-1]))
    print("RMSE: ", rmse(true_y_max, pred_y[:-1]))

    font = {'size': 30}
    import matplotlib

    matplotlib.rc('font', **font)

    import matplotlib.pyplot as plt
    plt.plot(true_x, true_y, 'b-', label='Readings')
    plt.plot(pred_x, pred_y, 'r-', label='LSTM-Online Predictions')
    df = "%A %d %B, %Y"
    plt.title("3002: Traffic Flow from {} to {}".format(true_x[0].strftime(df), true_x[-1].strftime(df)))
    plt.legend()

    plt.ylabel("Vehicles/ 5 min")
    plt.xlabel("Time")
    import os
    if os.getenv('DISPLAY', None):
        plt.show()
    else:
        out_img_name = model_name.split('/')[1][:-4]+'.png'
        plt.savefig(out_img_name)
