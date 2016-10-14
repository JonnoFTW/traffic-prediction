from collections import OrderedDict
from itertools import groupby

import pymongo
import tabulate
import os

if __name__ == "__main__":
    mongo_str = os.getenv('pymongo_conn', None)
    client = pymongo.MongoClient(mongo_str)
    results = client['mack0242']['hyperopt']
    headers = ['steps', 'geh', 'mape', 'rmse', 'dropout', 'batch_size', 'hidden', 'extra_layer', 'extra_layer_dropout',
               'loss function', 'optimiser', 'median_pe', 'mae', 'extra_layer_neurons']
    # opts = {h: True for h in headers}
    opts = {'_id': 0}
    trial_results = results.find({'steps':{'$ne':2}}, opts).sort("steps", 1)
    trial_results = [
        OrderedDict(sorted([(k, v) for k, v in row.items()], key=lambda x: headers.index(x[0]))) for row in
        trial_results
        ]
    print(
        tabulate.tabulate([min(g, key=lambda y: y['rmse']) for k, g in groupby(trial_results, lambda x: x['steps'])]
                          , headers='keys', tablefmt='latex'))
