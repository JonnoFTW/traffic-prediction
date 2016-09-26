#!/usr/bin/env python
import subprocess
import yaml

if __name__ == "__main__":
    with open('/home/mack0242/htm-models-adelaide/connection.yaml', 'r') as f:
        conf = yaml.load(f)
        pm_str = conf['mongo_uri']
    for i in [1, 3, 6, 9, 12]:
        print (subprocess.check_output(
            ['qsub', '-cwd', '-l', 'high_gpu=1', '-N', 'traffic_predict_step_' + str(i), '-o',
             'out_$JOB_NAME.$JOB_ID.log', '-e', 'error_$JOB_NAME.$JOB_ID.log', '-v',
             'site={},pymongo_conn={}'.format(i, pm_str),
             'job_runner.csh']))
