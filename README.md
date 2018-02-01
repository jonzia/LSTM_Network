# LSTM Network v1.1.2

## Overview
LSTM network implemented in Tensorflow designed for prediction and classification. **(4)**

## Description
This program is an LSTM network written in Python for Tensorflow. The architecture of the network is fully-customizable within the general framework, namely an LSTM network trained with a truncated BPTT algorithm where the output at each timestep is fed through a fully-connected layer to a variable number of outputs. The the input pipeline is a rolling-window with offset batches with customizable batch size and truncated BPTT length. The error is calculated via `tf.losses.mean_squared_error` and reduced via `tf.train.AdamOptimizer()`.

![Tensorboard Graph](LSTM_Graph.png)

## To Run
1. Install [Tensorflow](https://www.tensorflow.org/install/).
2. Download repository files.
  a. In *LSTM_main.py*, edit network characteristics in "User-Defined Parameters" field.
  ```python
BATCH_SIZE = 3		# Batch size
NUM_STEPS = 4		# Max steps for BPTT
NUM_LSTM_LAYERS = 1	# Number of LSTM layers
NUM_LSTM_HIDDEN = 5	# Number of LSTM hidden units
OUTPUT_UNITS = 1	# Number of FCL output units
INPUT_FEATURES = 9	# Number of input features
I_KEEP_PROB = 1.0	# Input keep probability / LSTM cell
O_KEEP_PROB = 1.0	# Output keep probability / LSTM cell
```
  b. Specify files to be used for network training and validation. (1)
 ```python
 with tf.name_scope("Training_Data"):
	tDataset = "file_path.csv"
with tf.name_scope("Validation_Data"):
	vDataset = "file_path.csv"
 ```
  c. Specify file directory for saving network variables.
 ```python
save_path = saver.save(sess, "file_path.csv")
 ```
3. Using the terminal or command window, run the python script *LSTM_main.py*. (2) (3)
4. (Optional) Analyze network parameters using [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

### What's New in v1.1.2
Added rolling-window input pipeline, saving network states, and LSTM cell dropout wrappers.

### Notes
**(1)** To use the stock input pipeline, the training datafiles must be CSV files with columns in the following arrangement:

TIMESTAMP | FEATURE_1 ... FEATURE_N | LABEL
----------|-------------------------|------
... | ... | ...

Note that the timestamp column is ignored by default and any column heading should be removed, as these may be read as input data. The number of feature columns may be variable and is set by the user-defined parameter `INPUT_FEATURES` at the top of the program.

**(2)** The program will output training loss, validation loss, and percent completion at each mini-batch.

**(3)** As of v1.1.1, ensure you have installed [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)!

**(4)** If you'd like to learn more about the design process and motivation for this program, feel free to check out my [blog](https://www.jonzia.me/projects/fog-problem)!

### References
(1) [Tensorflow's Recurrent Neural Network Tutorials](https://www.tensorflow.org/tutorials/recurrent)

(2) [Nicholas Locascio's LSTM Tutorial on GitHub](https://github.com/nicholaslocascio/bcs-lstm/blob/master/Lab.ipynb)

(3) [Colah's Blog: Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
