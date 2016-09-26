from csv import DictReader
from datetime import datetime
import numpy as np

def check_gpu():
    from theano import function
    import theano.tensor as T
    f = function([], T.exp([9]))
    if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')

def load_data(fname, EPS):
    docX = []
    print("Loading Data")
    with open(fname, 'r') as infile:
        reader = DictReader(infile)
        fields = reader.fieldnames
        for row in reader:
            dt = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
            counts = [int(row[x]) for x in fields[1:]]
            if any(map(lambda c: c > 300, counts)):
                # don't list those values that are extremely high
                continue
            x_row = [
                dt.weekday(),
                # is weekend
                int(dt.weekday() in [5, 6]),
                # hour of day
                dt.hour,
                dt.minute.real,
                max(1, sum(counts) + EPS)
            ]
            docX.append(x_row)
    print("Data loaded")
    return np.array(docX)


def train_test_split(x, y, test_size=0.33):
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must both have same number of rows")
    split_idx = int(x.shape[0] * (1 - test_size))
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]
