# LSTM Network v1.2.1

## Overview
LSTM network implemented in Tensorflow designed for time-series prediction and classification. Check out the [demo](https://youtu.be/DSzegLte0Iw) and [design blog](https://www.jonzia.me/projects/fog-problem)!

## Description
This program is an LSTM network written in Python for Tensorflow. The architecture of the network is fully-customizable within the general framework, namely an LSTM network trained with a truncated BPTT algorithm where the output at each timestep is fed through a fully-connected layer to a variable number of outputs. The the input pipeline is a rolling-window with offset batches with customizable batch size and truncated BPTT length. The error is calculated via `tf.nn.sigmoid_cross_entropy` and reduced via `tf.train.AdamOptimizer()`. This architecture was designed to solve time-series classification of data in one of two categories, though the model can be easily expanded in situations where there is more than one category.

Included with the LSTM program are Matlab and Python programs for processing .csv datasets and graphically analyzing the trained network. This package includes the UC Irvine Freezing of Gait data repository (converted to .csv file format) for which this program was designed as well as a *testdata.csv* file formatted for testing network function.

![Tensorboard Graph](https://github.com/jonzia/LSTM_Network/blob/master/Media/Graph120.PNG)

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
  c. (Optional) Pre-process data with *FeatureExtraction.m*. This program is designed to pre-process data from the attached *UC Irvine Freezing of Gait Dataset*, however, custom pre-processing code may be added in the data processing section.
  
  d. Specify file directory for saving network variables, summaries, and graph.
 ```python
with tf.name_scope("Model_Data"):	# Model save path
	save_path = "/tmp/model.ckpt"
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = ""
 ```
3. Using the terminal or command window, run the python script *LSTM_main.py*. (2) (3)
4. (Optional) Analyze network parameters using [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
5. (Optional) Run *test_bench.py* to analyze trained network outputs for a validation dataset. Ensure to select the correct filepaths for the validation dataset, and model load path, as well as the desired Filewriter save path and output .txt filenames. For proper analysis, also ensure that the user-defined parameters at the file header are the same as those used for training the network.
```python
with tf.name_scope("Validation_Data"):	# Validation dataset
	Dataset = ""
with tf.name_scope("Model_Data"):	# Model load path
	load_path = ".../tmp/model.ckpt"
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = ".../test_bench"
with tf.name_scope("Output_Data"):	# Output data filenames (.txt)
	# These .txt files will contain prediction and target data respectively for Matlab analysis
	prediction_file = ".../predictions.txt"
	target_file = ".../targets.txt"
```
6. (Optional) Run *NetworkAnalysis.m* to graphically analyze predictions and targets for the trained network and/or add custom in the data analysis section.

### Update Log
_v1.2.1_: Updated test bench to output predictions and targets to .txt file for Matlab analysis. Added Matlab program for analysis of the trained LSTM network.

_v1.2.0_: Updated mean-squared error loss approach to sigmoid cross-entropy. Added test bench program for running trained network on a validation dataset. Added feature extraction Matlab script for data pre-processing.

_v1.1.2_: Added ability to save network states and LSTM cell dropout wrappers.

_v1.1.1_: Added rolling-window input pipeline.

### Notes
**(1)** To use the stock input pipeline, the training datafiles must be CSV files with columns in the following arrangement:

TIMESTAMP | FEATURE_1 ... FEATURE_N | LABEL
----------|-------------------------|------
... | ... | ...

Note that the timestamp column is ignored by default and any column heading should be removed, as these may be read as input data. The number of feature columns may be variable and is set by the user-defined parameter `INPUT_FEATURES` at the top of the program.

**(2)** The program will output training loss, validation loss, percent completion and predictions/targets at each mini-batch.

**(3)** As of v1.1.1, ensure you have installed [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)!

### References
(1) [Tensorflow's Recurrent Neural Network Tutorials](https://www.tensorflow.org/tutorials/recurrent)

(2) [Nicholas Locascio's LSTM Tutorial on GitHub](https://github.com/nicholaslocascio/bcs-lstm/blob/master/Lab.ipynb)

(3) [Colah's Blog: Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
