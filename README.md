#LSTM Based Traffic Prediction

This project seeks to use LSTM for traffic prediction using the [Keras](https://keras.io/) frontend for [Theano](http://deeplearning.net/software/theano/). Hyperparameter optimisation is used to find the best set of parameters for the network, which I found to be:

###Usage

Run:

```
pip install -r requirements.txt
```

Then edit [main.py](https://github.com/JonnoFTW/traffic-prediction/blob/master/main.py) so that it uses your own .csv file. It will attempt to store hyperparameter results in mongodb. You can use [show_results.py](https://github.com/JonnoFTW/traffic-prediction/blob/master/show_results.py) to view them.

Scripts are also provided to speed up processing via Sun Grid Engine. Though they are unreliable because of the length of time it takes for theano to compile the models. You will also need CUDA and a compatible card if you want the speedups they provide. 