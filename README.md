# LSTM Network in Tensorflow

## Overview
LSTM network implemented in Tensorflow designed for prediction and classification.

## Description:
This program is an LSTM network written in Python for Tensorflow. The architecture of the network is fully-customizable within the general framework, namely an LSTM network trained with a truncated BPTT algorithm where the output at each timestep is fed through a fully-connected layer to a variable number of outputs. The error is calculated via `tf.losses.mean_squared_error` and reduced via `tf.train.AdamOptimizer()`.

## To Run:
1. Install [Tensorflow](https://www.tensorflow.org/install/).
2. Download repository files.
  a. In *LSTM_main.py*, edit network characteristics in "User-Defined Parameters" field.
  b. Specify files to be used for network training and validation. (1)
3. Using the terminal or command window, run the python script *LSTM_main.py*. (2)
4. (Optional) Analyze network parameters using [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).



### What's New in v1.1.0
The current version is the first fully-functional version of the LSTM program. Check back for future updates!

### Notes
**(1)** To use the stock input pipeline, the training datafiles must be CSV files with columns in the following arrangement:

*TIMESTAMP | FEATURE_1 ... FEATURE_N | LABEL*

Note that the timestamp column is ignored by default and any column heading should be removed, as these may be read as input data. The number of feature columns may be variable and is set by the user-defined parameter `INPUT_FEATURES` at the top of the program.

**(2)** The program will output loss for training and validation at each mini-batch.

### References
(1) [Tensorflow's Recurrent Neural Network Tutorials](https://www.tensorflow.org/tutorials/recurrent)
(2) [Nicholas Locascio's LSTM Tutorial on GitHub](https://github.com/nicholaslocascio/bcs-lstm/blob/master/Lab.ipynb)
