# LSTM Network v1.3.1

## Overview
LSTM network implemented in Tensorflow designed for time-series prediction and classification. Check out the [demo](https://youtu.be/DSzegLte0Iw) and [design blog](https://www.jonzia.me/projects/fog-problem)!

## Description
This program is an LSTM network written in Python for Tensorflow. The architecture of the network is fully-customizable within the general framework, namely an LSTM network trained with a truncated BPTT algorithm where the output at each timestep is fed through a fully-connected layer to a variable number of outputs. The the input pipeline can be either rolling-window with offset batches or balanced minibatches and weights updated via truncated BPTT. The error is calculated via `tf.nn.softmax_cross_entropy_with_logits_v2` and reduced via `tf.train.AdamOptimizer()`. This architecture was designed to solve time-series classification of data in two or more categories, either one-hot or binary-encoded.

Included with the LSTM program are Matlab and Python programs for processing .csv datasets and graphically analyzing the trained network. *FeatureExtraction.m* contains code designed for pre-processing the input data (including low-pass filtering, RMS, and Fourier analysis), however the data processing section may be adapted for any purpose. *NetworkAnalysis.m*, contains code designed for visualizing network performance for 3 (one-hot) to 8 (binary-encoded) categories of classification and classifying outputs via Matlab's clasificationLearner, though it may also be adapted for any purpose.

![Tensorboard Graph](https://raw.githubusercontent.com/jonzia/LSTM_Network/master/Media/Graph122.PNG)

## To Run
1. Install [Tensorflow](https://www.tensorflow.org/install/).
2. Download repository files.
  a. In *LSTM_main.py*, edit network characteristics in "User-Defined Parameters" field. (1)
  ```python
BATCH_SIZE = 3		# Batch size
NUM_STEPS = 100		# Max steps for BPTT
NUM_LSTM_LAYERS = 1	# Number of LSTM layers
NUM_LSTM_HIDDEN = 5	# Number of LSTM hidden units
OUTPUT_CLASSES = 3	# Number of classes / FCL output units
INPUT_FEATURES = 9	# Number of input features
I_KEEP_PROB = 1.0	# Input keep probability / LSTM cell
O_KEEP_PROB = 1.0	# Output keep probability / LSTM cell
NUM_TRAINING = 10000	# Number of training batches (balanced minibatches)
NUM_VALIDATION = 100	# Number of validation batches (balanced minibatches)
WINDOW_INT_t = 1	# Rolling window step interval for training (rolling window)
WINDOW_INT_v = 1000	# Rolling window step interval for validation (rolling window)
LOAD_FILE = False 	# Load initial LSTM model from saved checkpoint?
```
  b. Specify files to be used for network training and validation. (2)
 ```python
 with tf.name_scope("Training_Data"):
	tDataset = "file_path.csv"
with tf.name_scope("Validation_Data"):
	vDataset = "file_path.csv"
 ```
  c. (Optional) Pre-process data with *FeatureExtraction.m*. Custom pre-processing code may be added in the data processing section.
  
  d. Specify file directory for saving network variables, summaries, and graph.
 ```python
with tf.name_scope("Model_Data"):	# Model save path
	save_path = "/tmp/model"
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = "/output"
 ```
3. Using the terminal or command window, run the python script *LSTM_main.py*. (3) (4) (5)
4. (Optional) Analyze network parameters using [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
5. (Optional) Run *test_bench.py* to analyze trained network outputs for a validation dataset (predictions and targets saved as .txt files). Ensure to select the correct filepaths for the validation dataset, and model load path, as well as the desired Filewriter save path and output .txt filenames. For proper analysis, also ensure that the user-defined parameters at the file header are the same as those used for training the network.
```python
with tf.name_scope("Validation_Data"):	# Validation dataset
	Dataset = ""
with tf.name_scope("Model_Data"):	# Model load path
	load_path = ".../tmp/model"
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = ".../test_bench"
with tf.name_scope("Output_Data"):	# Output data filenames (.txt)
	# These .txt files will contain prediction and target data respectively for Matlab analysis
	prediction_file = ".../predictions.txt"
	target_file = ".../targets.txt"
```
6. (Optional) Run *NetworkAnalysis.m* to graphically analyze predictions and targets for the trained network and/or add custom in the data analysis section. Data may be fed into Matlab's classificationLearner to update the classification rule from the simple `round(predictions)` example provided.

![Example NetworkAnalysis.m Output](https://raw.githubusercontent.com/jonzia/LSTM_Network/master/Media/ExamplePredictionAnalysis.png)

### Update Log
_v1.3.1_: Introduced balanced minibatch input pipeline. Bug fixes and improvements.

_v1.2.1_ - _v1.2.4_: Improved exception handling and added ability to load partially-trained networks to resume training. Added classification learning capability for trained network outputs and ability to conditionally call optimizer when running the session to mitigate class imbalance. Added capability for >2 target categories (vs. binary classification) and rolling window step interval for decreasing training time overfitting likelihood. Updated test bench to output predictions and targets to .txt file for Matlab analysis. Added Matlab program for analysis of the trained LSTM network.

_v1.1.1_ - _v1.2.0_: Updated mean-squared error loss approach to sigmoid cross-entropy. Added test bench program for running trained network on a validation dataset. Added feature extraction Matlab script for data pre-processing. Added ability to save network states and LSTM cell dropout wrappers. Added rolling-window input pipeline.

### Notes
**(1)** To select rolling-window input pipeline, swap hashes on lines 227/228, 254/255, and 267/268:
```python
features, labels = extract_data_balanced(tDataset, BATCH_SIZE, NUM_STEPS, INPUT_FEATURES, OUTPUT_CLASSES, file_length) # Balanced minibatches
# features, labels = extract_data_window(tDataset, BATCH_SIZE, NUM_STEPS, INPUT_FEATURES, OUTPUT_CLASSES, step) # Rolling window
...
v_features, v_labels =  extract_data_balanced(vDataset, BATCH_SIZE, NUM_STEPS, INPUT_FEATURES, OUTPUT_CLASSES, v_file_length) # Balanced minibatches
# v_features, v_labels =  extract_data_window(vDataset, BATCH_SIZE, NUM_STEPS, INPUT_FEATURES, OUTPUT_CLASSES, step_num) # Rolling window
...
p_completion = 100*step/NUM_TRAINING
# p_completion = 100*step/file_length
```

**(2)** To use the stock input pipeline, the training datafiles must be CSV files with columns in the following arrangement:

TIMESTAMP | FEATURE_1 ... FEATURE_N | LABEL_1 ... LABEL_M
----------|-------------------------|------
... | ... | ...

Note that the timestamp column is ignored by default and any column heading should be removed, as these may be read as input data. The number of feature and label columns may be variable and is set by the user-defined parameters `INPUT_FEATURES` and `OUTPUT_CLASSES` respectively at the top of the program.

**(3)** The program will output training loss, validation loss, percent completion and predictions/targets at each mini-batch.

**(4)** As of v1.1.1, ensure you have installed [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)!

**(5)** As of v1.2.3, ensure that you have the most recent version of Tensorflow installed (1.5).

### References
(1) [Tensorflow's Recurrent Neural Network Tutorials](https://www.tensorflow.org/tutorials/recurrent)

(2) [Nicholas Locascio's LSTM Tutorial on GitHub](https://github.com/nicholaslocascio/bcs-lstm/blob/master/Lab.ipynb)

(3) [Colah's Blog: Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
