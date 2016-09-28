#!/usr/bin/env python
import subprocess
import os

if __name__ == "__main__":
    pm_str = os.getenv('pymongo_conn', None)
    if not pm_str:
        exit("please provide a pymongo_conn as an env variable")
    for i in [1, 3, 6, 9, 12]:
        print (subprocess.check_output(
            ['qsub', '-cwd', '-l', 'high_gpu=1', '-N', 'traffic_predict_step_' + str(i), '-o',
             'out_$JOB_NAME.$JOB_ID.log', '-e', 'error_$JOB_NAME.$JOB_ID.log', '-v',
             'site={},pymongo_conn={}'.format(i, pm_str),
             'job_runner.csh']))
