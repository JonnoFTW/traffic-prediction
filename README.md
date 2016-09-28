#LSTM Based Traffic Prediction

This project seeks to use LSTM for traffic prediction using the [Keras](https://keras.io/) frontend for [Theano](http://deeplearning.net/software/theano/). Hyperparameter optimisation is used to find the best set of parameters for the network, which I found to be:

###Usage

Run:

```
pip install -r requirements.txt
```

Then edit [main.py](https://github.com/JonnoFTW/traffic-prediction/blob/master/main.py) so that it uses your own parameters for the network. It will attempt to store hyperparameter results in mongodb. You can use [show_results.py](https://github.com/JonnoFTW/traffic-prediction/blob/master/show_results.py) to view them.

Run using:
```
python main.py 1 /path/to/some.csv
```

The CSV should have a the following format:
```
timestamp,16,17,18,19,20,21
2011-12-31 23:55:00,4,6,8,13,3,0
2012-01-01 00:00:00,5,7,8,10,3,2
2012-01-01 00:05:00,3,3,7,15,1,2
2012-01-01 00:10:00,9,3,3,7,1,1
2012-01-01 00:15:00,3,4,12,15,2,0
2012-01-01 00:20:00,7,5,11,19,2,2
```
Basically, the first column must be a timestamp and the rest integer values.


Scripts are also provided to speed up processing via Sun Grid Engine. Though they are unreliable because of the length of time it takes for theano to compile the models. You will also need CUDA and a compatible card if you want the speedups they provide. 