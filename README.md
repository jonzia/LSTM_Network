# LSTM Network v1.3.3

## Overview
LSTM network implemented in Tensorflow designed for time-series prediction and classification. Check out the [demo](https://youtu.be/DSzegLte0Iw) and [design blog](https://www.jonzia.me/projects/lstm-tensorflow)!

## Description
This program is an LSTM network written in Python for Tensorflow. The architecture of the network is fully-customizable within the general framework, namely an LSTM network trained with a truncated BPTT algorithm where the output at each timestep is fed through a fully-connected layer to a variable number of outputs. The the input pipeline can be either rolling-window with offset batches or balanced minibatches and weights updated via truncated BPTT. The error is calculated via `tf.nn.softmax_cross_entropy_with_logits_v2` and reduced via `tf.train.AdamOptimizer()`. This architecture was designed to solve time-series classification of data in two or more categories (one-hot encoded).

Included with the LSTM program are Matlab and Python programs for processing .csv datasets and graphically analyzing the trained network. *FeatureExtraction.m* contains code designed for pre-processing the input data (including low-pass filtering, RMS, and Fourier analysis), however the data processing section may be adapted for any purpose. *NetworkAnalysis.m*, contains code designed for visualizing network performance for 3 (one-hot) to 8 (binary-encoded) categories of classification and classifying outputs via Matlab's clasificationLearner, though it may also be adapted for any purpose.

![Tensorboard Graph](https://raw.githubusercontent.com/jonzia/LSTM_Network/master/Media/Graph122.PNG)

## Features
- Fully-customizable network architecture and training setup.
- Input data processing and trained network validation and analysis code included.
- Ability to save and load checkpoints to pause and continue training.
- Balanced minibatching and rolling-window input pipelines included.
- Ability to visualize training with Tensorboard and write data to output files for analysis.
- Modular, clean, and commented implementation -- highly adaptable for any purpose!
- Plug and Play: It just works.

## To Run
1. Install [Tensorflow](https://www.tensorflow.org/install/).
2. Download repository files.
  a. In *network.py*, set the network architecture used in the training and test bench programs.
  ```python
# Architecture
self.batch_size = 9		# Batch size
self.num_steps = 100		# Max steps for BPTT
self.num_lstm_layers = 1	# Number of LSTM layers
self.num_lstm_hidden = 10	# Number of LSTM hidden units
self.output_classes = 3		# Number of classes / FCL output units
self.input_features = 9		# Number of input features
  ```
  b. In *LSTM_Main.py*, set training parameters.
  ```python
# Training
I_KEEP_PROB = 1.0		# Input keep probability / LSTM cell
O_KEEP_PROB = 1.0		# Output keep probability / LSTM cell
NUM_TRAINING = 1000		# Number of training batches (balanced minibatches)
NUM_VALIDATION = 100		# Number of validation batches (balanced minibatches)
WINDOW_INT_t = 1		# Rolling window step interval for training (rolling window)
WINDOW_INT_v = 1000		# Rolling window step interval for validation (rolling window)
# Load File
LOAD_FILE = False 	# Load initial LSTM model from saved checkpoint?
# Input Pipeline
# Enter "True" for balanced mini-batching and "False" for rolling-window
MINI_BATCH = True
  ```
  c. Specify files to be used for network training and validation. **(1)**
 ```python
# Specify filenames
# Root directory:
dir_name = "/Directory"
with tf.name_scope("Training_Data"):	# Training dataset
	tDataset = os.path.join(dir_name, "data/filename.csv")
with tf.name_scope("Validation_Data"):	# Validation dataset
	vDataset = os.path.join(dir_name, "data/filename")
 ```
  d. (Optional) Pre-process data with *FeatureExtraction.m*. Custom pre-processing code may be added in the data processing section.
  
  e. Specify file directory for saving network variables, summaries, and graph.
 ```python
with tf.name_scope("Model_Data"):		# Model save path
	save_path = os.path.join(dir_name, "checkpoints/model")		# Save model at each step
	save_path_op = os.path.join(dir_name, "checkpoints/model_op")	# Save optimal model
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = os.path.join(dir_name, "output")
 ```
3. Using the terminal or command window, run the python script *LSTM_main.py*. **(2) (3) (4)**
4. (Optional) Analyze network parameters using [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
5. (Optional) Run *test_bench.py* to analyze trained network outputs for a validation dataset (predictions and targets saved as .txt files). Ensure to select the correct filepaths for the validation dataset, and model load path, as well as the desired Filewriter save path and output .txt filenames. For proper analysis, also ensure that the user-defined parameters at the file header are the same as those used for training the network.
```python
dir_name = "/Users/jonathanzia/Dropbox/Documents/Projects/TensorFlow"
with tf.name_scope("Training_Data"):	# Testing dataset
	Dataset = os.path.join(dir_name, "UC Irvine Dataset/dataset/S01R02_lpf.csv")
with tf.name_scope("Model_Data"):		# Model load path
	load_path = os.path.join(dir_name, "checkpoints/model")
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = os.path.join(dir_name, "output")
with tf.name_scope("Output_Data"):		# Output data filenames (.txt)
	# These .txt files will contain prediction and target data respectively for Matlab analysis
	prediction_file = os.path.join(dir_name, "predictions.txt")
	target_file = os.path.join(dir_name, "targets.txt")
```
6. (Optional) Run *NetworkAnalysis.m* to graphically analyze predictions and targets for the trained network and/or add custom code in the data analysis section. When the program is run for the first time, data may be fed into Matlab's classificationLearner, from which the desired classifier (_trainedClassifier.mat_) can be generated and applied to subsequent tests.
```matlab
trainClass = true;  % true = train classifier, false = load model
```
A trained classification model can then be loaded at a later point. Simply select save and load paths for the model.
```matlab
save('../trainedModel.mat','trainedModel'); % Save model
...
load('../trainedModel.mat'); % Load model
```

![Example NetworkAnalysis.m Output](https://raw.githubusercontent.com/jonzia/LSTM_Network/master/Media/ExamplePredictionAnalysis.png)

### Update Log
_v1.3.3_: Added ability to save optimal model state as checkpoint. Included *network.py* so that network architecture is preserved across all python files. Bug fixes and improvements.

_v1.3.1_ - _v1.3.2_: Improved user interface and ability to handle a variable number of output classes as one-hot vectors. Introduced balanced minibatch input pipeline. Bug fixes and improvements.

_v1.2.1_ - _v1.2.4_: Improved exception handling and added ability to load partially-trained networks to resume training. Added classification learning capability for trained network outputs and ability to conditionally call optimizer when running the session to mitigate class imbalance. Added capability for >2 target categories (vs. binary classification) and rolling window step interval for decreasing training time overfitting likelihood. Updated test bench to output predictions and targets to .txt file for Matlab analysis. Added Matlab program for analysis of the trained LSTM network.

_v1.1.1_ - _v1.2.0_: Updated mean-squared error loss approach to sigmoid cross-entropy. Added test bench program for running trained network on a validation dataset. Added feature extraction Matlab script for data pre-processing. Added ability to save network states and LSTM cell dropout wrappers. Added rolling-window input pipeline.

### Notes
**(1)** To use the stock input pipeline, the training datafiles must be CSV files with columns in the following arrangement:

TIMESTAMP | FEATURE_1 ... FEATURE_N | LABEL_1 ... LABEL_M
----------|-------------------------|------
... | ... | ...

Note that the timestamp column is ignored by default and any column heading should be removed, as these may be read as input data. The number of feature and label columns may be variable and is set by the user-defined parameters `INPUT_FEATURES` and `OUTPUT_CLASSES` respectively at the top of the program. The labels should be a one-hot vector when using the provided softmax loss function.

**(2)** The program will output training loss, validation loss, percent completion and predictions/targets at each mini-batch.

**(3)** As of v1.1.1, ensure you have installed [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)!

**(4)** As of v1.2.3, ensure that you have the most recent version of Tensorflow installed (1.5).

### References
(1) [Tensorflow's Recurrent Neural Network Tutorials](https://www.tensorflow.org/tutorials/recurrent)

(2) [Nicholas Locascio's LSTM Tutorial on GitHub](https://github.com/nicholaslocascio/bcs-lstm/blob/master/Lab.ipynb)

(3) [Colah's Blog: Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
