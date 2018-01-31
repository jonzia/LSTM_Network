# ----------------------------------------------------
# LSTM Network Implementation using Tensorflow 1.1.1
# Created by: Jonathan Zia
# Last Modified: Friday, Jan 26, 2018
# Georgia Institute of Technology
# ----------------------------------------------------
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import csv


# ----------------------------------------------------
# User-Defined Constants
# ----------------------------------------------------
BATCH_SIZE = 5			# Batch size
NUM_STEPS = 6			# Max steps for BPTT
NUM_LSTM_LAYERS = 1		# Number of LSTM layers
NUM_LSTM_HIDDEN = 5		# Number of LSTM hidden units
OUTPUT_UNITS = 1		# Number of FCL output units
INPUT_FEATURES = 9		# Number of input features

# ----------------------------------------------------
# Input data files
# ----------------------------------------------------
# Specify filenames
with tf.name_scope("Training_Data"):
	tDataset = "D:\\Dropbox\\Documents\\Projects\\TensorFlow\\UC Irvine Dataset\\dataset\\testdata.csv"
with tf.name_scope("Validation_Data"):
	vDataset = "D:\\Dropbox\\Documents\\Projects\\TensorFlow\\UC Irvine Dataset\\dataset\\testdata.csv"

# Obtain length of testing and validation datasets
file_length = len(pd.read_csv(tDataset))
v_file_length = len(pd.read_csv(vDataset))


# ----------------------------------------------------
# User-Defined Methods
# ----------------------------------------------------

def init_values(shape):
	"""
	Initialize Weight and Bias Matrices
	Returns: Tensor of shape "shape" w/ normally-distributed values
	"""
	temp = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(temp)

def extract_data(filename, batch_size, num_steps, input_features, epoch):
	"""
	Extract features and labels from filename.csv in rolling-window batches
	Returns:
	feature_batch ~ [batch_size, num_steps, input_features]
	label_batch ~ [batch_size, num_steps]
	"""

	# Initialize numpy arrays for return value placeholders
	feature_batch = np.zeros((batch_size,num_steps,input_features))
	label_batch = np.zeros((batch_size,num_steps))

	# Import data from CSV as a sliding window:
	# First, import data starting from t = epoch to t = epoch + num_steps
	# ... add feature data to feature_batch[0, :, :]
	# ... add label data to label_batch[batch, :, :]
	# Then, import data starting from t = epoch + 1 to t = epoch + num_steps + 1
	# ... add feature data to feature_batch[1, :, :]
	# ... add label data to label_batch[batch, :, :]
	# Repeat for all batches.
	for i in range(batch_size):
		temp = pd.read_csv(filename, skiprows=epoch*batch_size+i, nrows=num_steps, header=None)
		temp = temp.as_matrix()
		feature_batch[i,:,:] = temp[:,1:input_features+1]
		label_batch[i,:] = temp[:,input_features+1]

	# Return feature and label batches
	return feature_batch, label_batch


# ----------------------------------------------------
# Importing FoG Dataset Batches
# ----------------------------------------------------
# Create placeholders for inputs and target values
# Input dimensions: BATCH_SIZE x NUM_STEPS x INPUT_FEATURES
# Target dimensions: BATCH_SIZE x NUM_STEPS (x 1)
inputs = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, INPUT_FEATURES], name="Input_Placeholder")
targets = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS], name="Target_Placeholder")


# ----------------------------------------------------
# Building a Multilayer LSTM
# ----------------------------------------------------
# Build LSTM cells
with tf.name_scope("LSTM_Network"):
	cells = []
	for _ in range(NUM_LSTM_LAYERS):
		cell = tf.contrib.rnn.BasicLSTMCell(NUM_LSTM_HIDDEN)
		cells.append(cell)
	stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

# Initialize weights and biases for fully-connected layer.
with tf.name_scope("FCN_Variables"):
	W = init_values([NUM_LSTM_HIDDEN, OUTPUT_UNITS])
	tf.summary.histogram('Weights',W)
	b = init_values([OUTPUT_UNITS])
	tf.summary.histogram('Biases',b)

# Add LSTM cells to dynamic_rnn and implement truncated BPTT
initial_state = state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
logits = []
with tf.name_scope("Dynamic_RNN"):
	for i in range(NUM_STEPS):
		# Obtain output at each step
		output, state = tf.nn.dynamic_rnn(stacked_lstm, inputs[:,i:i+1,:], initial_state=state)
		# Reshape output to remove extra dimension
		output = tf.reshape(output,[BATCH_SIZE,NUM_LSTM_HIDDEN])

		with tf.name_scope("Append_Output"):
			# Obtain logits by passing output
			for j in range(BATCH_SIZE):
				logit = tf.nn.sigmoid(tf.matmul(output[j:j+1,:], W) + b)
				logits.append(logit)
# Converting logits array to tensor and reshaping the array such that it has the arrangement:
# [time1 time2] batch = 1
# [time1 time2] batch = 2 ...
with tf.name_scope("Calculate_Logits"):
	logits = tf.reshape(tf.convert_to_tensor(logits), [BATCH_SIZE, NUM_STEPS])

# ----------------------------------------------------
# Calculate Loss and Define Optimizer
# ----------------------------------------------------
# Calculating mean squared error of labels and logits
loss = tf.losses.mean_squared_error(targets, logits)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
init = tf.global_variables_initializer()
with tf.Session() as sess:
	# Create Tensorboard graph
	writer = tf.summary.FileWriter("output", sess.graph)
	merged = tf.summary.merge_all()
	# Initialize the variables
	sess.run(init)

	# Start populating the filename queue
	# Coordinator: Coordinates the termination of a set of threads
	# Start_queue_runners: Starts all queue runners in graph
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	# Training the network
	# Set range (prevent index out-of-range exception for rolling window)
	rng = math.floor((file_length - BATCH_SIZE - NUM_STEPS + 2) / BATCH_SIZE)
	for step in range(rng):

		# Obtaining batch of features and labels from TRAINING dataset(s)
		features, labels = extract_data(tDataset, BATCH_SIZE, NUM_STEPS, INPUT_FEATURES, step)

		# Input data
		data = {inputs: features, targets:labels}
		# Run optimizer, loss, and predicted error ops in graph
		_, loss_train = sess.run([optimizer,loss], feed_dict=data)

		# Evaluate network and print data in terminal periodically
		with tf.name_scope("Validation"):
			if step % 50 == 0:
				print("Minibatch train loss at step", step, ":", loss_train)

				# Evaluate network
				test_loss = []
				# Set range (prevent index out-of-range exception for rolling window)
				v_rng = math.floor((v_file_length - BATCH_SIZE - NUM_STEPS + 2) / BATCH_SIZE)
				for step_num in range(v_rng):

					# Obtaining batch of features and labels from VALIDATION dataset(s)
					v_features, v_labels =  extract_data(vDataset, BATCH_SIZE, NUM_STEPS, INPUT_FEATURES, step_num)

					# Input data and run session to find loss
					data_test = {inputs: v_features, targets: v_labels}
					loss_test = sess.run(loss, feed_dict=data_test)
					test_loss.append(loss_test)
				print("Test loss: %.3f" % np.mean(test_loss))

		# Writing summaries to Tensorboard
		summ = sess.run(merged)
		writer.add_summary(summ,step)
	# Request_Stop: Request that the threads stop
	# Join: Wait for threads to terminate
	coord.request_stop()
	coord.join(threads)
	# Close the writer
	writer.close()
