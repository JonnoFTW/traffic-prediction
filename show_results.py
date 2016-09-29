from collections import OrderedDict
from itertools import groupby

import pymongo
import tabulate
import os


if __name__ == "__main__":
    mongo_str = os.getenv('pymongo_conn', None)
    client = pymongo.MongoClient(mongo_str)
    results = client['mack0242']['hyperopt']
    headers = ['steps', 'geh', 'mape', 'rmse', 'dropout', 'batch_size', 'hidden', 'extra_layer', 'extra_layer_dropout']
    opts = {h: True for h in headers}
    opts['_id'] = 0
    trial_results = results.find({}, opts)
    trial_results = [
        OrderedDict(sorted([(k, v) for k, v in row.items()], key=lambda x: headers.index(x[0]))) for row in trial_results
        ]
    print(
    tabulate.tabulate([min(g, key=lambda y:y['geh']) for k,g in groupby(trial_results, lambda x: x['steps'])]
                      , headers='keys', tablefmt='latex'))
