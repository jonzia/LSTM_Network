# LSTM Network v1.1.3

## Overview
LSTM network implemented in Tensorflow designed for prediction and classification. **(4)**

## Description
This program is an LSTM network written in Python for Tensorflow. The architecture of the network is fully-customizable within the general framework, namely an LSTM network trained with a truncated BPTT algorithm where the output at each timestep is fed through a fully-connected layer to a variable number of outputs. The the input pipeline is a rolling-window with offset batches with customizable batch size and truncated BPTT length. The error is calculated via `tf.nn.sigmoid_cross_entropy` and reduced via `tf.train.AdamOptimizer()`. This architecture was designed to solve time-series classification of data in one of two categories, though the model can be easily expanded in situations where there is more than one category.

![Tensorboard Graph](https://github.com/jonzia/LSTM_Network/blob/master/Media/Graph113.PNG)

## To Run
1. Install [Tensorflow](https://www.tensorflow.org/install/).
2. Download repository files.
  a. In *LSTM_main.py*, edit network characteristics in "User-Defined Parameters" field.
  ```python
BATCH_SIZE = 3		# Batch size
NUM_STEPS = 5		# Max steps for BPTT
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
  c. Specify file directory for saving network variables, summaries, and graph.
 ```python
with tf.name_scope("Model_Data"):	# Model save path
	save_path = "/tmp/model.ckpt"
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = ""
 ```
3. Using the terminal or command window, run the python script *LSTM_main.py*. (2) (3)
4. (Optional) Analyze network parameters using [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
5. Run *test_bench_Rev02.py* to analyze trained network outputs for a validation dataset. Ensure to select the correct filepaths for the validation dataset, model load path, and Filewriter save path.
```python
with tf.name_scope("Training_Data"):	# Testing dataset
	Dataset = ""
with tf.name_scope("Model_Data"):	# Model load path
	load_path = "model.ckpt"
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = "test_bench"
```

### Update Log
_v1.1.3_: Updated mean-squared error loss approach to sigmoid cross-entropy. Added test bench program for running trained network on a validation dataset.

_v1.1.2_: Added ability to save network states and LSTM cell dropout wrappers.

_v1.1.1_: Added rolling-window input pipeline.

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
