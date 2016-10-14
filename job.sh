#!/bin/bash
source ~/.bashrc
source  ~/.bash_profile
export PATH="$HOME/.pyenv/bin:$PATH:/home/mack0242/tools/bin"
pyenv local 2.7.12
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
export LD_LIBRARY_PATH=/home/mack0242/tools/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:/home/mack0242/tools/lib:$LD_LIBRARY_PATH
echo "Running traffic predict"

declare -a arr=("1" "3" "6" "9" "12")
for i in "${arr[@]}"
do
    echo "Starting step $i"
    python /home/mack0242/traffic-prediction/main.py $i lane_data.csv
done

