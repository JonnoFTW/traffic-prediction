#!/bin/bash
source  ~/.bash_profile
pyenv local 2.7.12
export PATH="$HOME/.pyenv/bin:$PATH:/home/mack0242/tools/bin"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
export LD_LIBRARY_PATH=/home/mack0242/tools/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:/home/mack0242/tools/lib:$LD_LIBRARY_PATH
echo "Running traffic predict at $steps steps"
python /home/mack0242/traffic-prediction/main.py $steps lane_data.csv