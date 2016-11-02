from csv import DictReader
from datetime import datetime, date
import numpy as np
from keras.callbacks import Callback
from pymongo import MongoClient


def load_mongo_data(fromNode, toNode):
    import os
    mongo_uri = os.getenv('mongo_conn', None)
    if mongo_uri is None:
        exit("Please set mongo_conn environment variable")
    client = MongoClient(mongo_uri)
    locations_collection = client['mack0242']['locations']
    readings_collection = client['mack0242']['readings']
    aNode, bNode = locations_collection.find({'intersection_number': {'$in':[fromNode, toNode]},
                                                  'neighbours_sensors': {'$exists': True}})
    if aNode['intersection_number'] == fromNode and bNode['intersection_number'] == toNode:
        fromNode = aNode
        toNode = bNode
    else:
        fromNode = bNode
        toNode = aNode

    # load data into a numpy array
    readings = readings_collection.find({'site_no': toNode})
    sensors = fromNode['neighbours_sensors'][toNode]['to']
    docs = []
    for row in readings:
        dt = row['datetime']
        docs.append([
            dt.weekday(),
            dt.isocalendar()[1],
            int(dt.weekday() in [5, 6]),
            dt.hour,
            dt.minute,
            max(1, sum(row['readings'].values()))
        ])
    return np.array(docs)


def check_gpu():
    from theano import function
    import theano.tensor as T
    f = function([], T.exp([9]))
    if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')
def load_holidays():
    dates =[]
    return set(dates)
def load_data(fname, EPS, use_datetime=False, load_from=None, limit=np.inf, use_sensors=None, end_date=None):
    docX = []
    print("Loading Data")
    rows = 0
    holidays = load_holidays()
    with open(fname, 'r') as infile:
        reader = DictReader(infile)
        fields = reader.fieldnames
        for row in reader:
            rows += 1
            if rows > limit:
                break
            dt = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
            if end_date is not None and dt >= end_date:
                break
            if load_from is not None and dt <= load_from:
                continue
            if type(use_sensors) is list:
                counts = [int(row[x]) for x in fields[1:] if int(x) in use_sensors]
            else:
                counts = [int(row[x]) for x in fields[1:]]
            if any(map(lambda c: c > 300, counts)) and not use_datetime:
                # don't list those values that are extremely high
                continue
            if not use_datetime:
                x_row = [
                    dt.weekday(),
                    # is weekend
                    int(dt.weekday() in [5, 6]),
                    # week of year
                    # dt.isocalendar()[1],
                    #is holiday
                    # int(dt in holidays),
                    # hour of day
                    dt.hour,
                    dt.minute,
                    max(1, sum(counts) + EPS)
                ]
            else:
                x_row = [
                    dt, sum(counts)
                ]
            docX.append(x_row)
    print("Data loaded")
    return np.array(docX)


def train_test_split(x, y, test_size=0.33):
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must both have same number of rows")
    split_idx = int(x.shape[0] * (1 - test_size))
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]


class ResetStatesCallback(Callback):
    def __init__(self, max_len=20):
        self.counter = 0
        self.max_len = max_len

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            print("Resetting states")
            self.model.reset_states()
        self.counter += 1


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

