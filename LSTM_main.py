# ----------------------------------------------------
# LSTM Network Implementation using Tensorflow 1.3.3
# Created by: Jonathan Zia
# Last Modified: Wednesday, Feb 14, 2018
# Georgia Institute of Technology
# ----------------------------------------------------
import tensorflow as tf
import network as net
import pandas as pd
import random as rd
import numpy as np
import time
import math
import csv
import os


# ----------------------------------------------------
# Instantiate Network Class
# ----------------------------------------------------
lstm = net.Network()


# ----------------------------------------------------
# User-Defined Constants
# ----------------------------------------------------
# Training
I_KEEP_PROB = 1.0		# Input keep probability / LSTM cell
O_KEEP_PROB = 1.0		# Output keep probability / LSTM cell
NUM_TRAINING = 1000		# Number of training batches (balanced minibatches)
NUM_VALIDATION = 100	# Number of validation batches (balanced minibatches)
WINDOW_INT_t = 1		# Rolling window step interval for training (rolling window)
WINDOW_INT_v = 1000		# Rolling window step interval for validation (rolling window)

# Load File
LOAD_FILE = False 	# Load initial LSTM model from saved checkpoint?

# Input Pipeline
# Enter "True" for balanced mini-batching and "False" for rolling-window
MINI_BATCH = True


# ----------------------------------------------------
# Input data files
# ----------------------------------------------------
# Specify filenames
# Root directory:
dir_name = "/Directory"
with tf.name_scope("Training_Data"):	# Training dataset
	tDataset = os.path.join(dir_name, "data/filename.csv")
with tf.name_scope("Validation_Data"):	# Validation dataset
	vDataset = os.path.join(dir_name, "data/filename")
with tf.name_scope("Model_Data"):		# Model save path
	save_path = os.path.join(dir_name, "checkpoints/model")		# Save model at each step
	save_path_op = os.path.join(dir_name, "checkpoints/model_op")	# Save optimal model
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = os.path.join(dir_name, "output")

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

def extract_data_balanced(filename, batch_size, num_steps, input_features, output_classes, f_length):
	"""
	Extract features and labels from filename.csv in class-balanced batches
	Returns:
	feature_batch ~ [batch_size, num_steps, input_features]
	label_batch ~ [batch_size, output_classes]
	"""

	# Initialize numpy arrays for return value placeholders
	feature_batch = np.zeros((batch_size,num_steps,input_features))
	label_batch = np.zeros((batch_size, output_classes))

	# Import data from CSV as a balanced minibatch:
	# First, select datapoint from class i by generating random integer in range [0, f_length-num_steps-1]
	# Second, check that the label at num_steps belongs to one of the three classes
	# If the label belongs to the proper class, add data to batch and increment i
	# Continue for all classes
	# Define classes:
	classes = np.identity(output_classes) # Returns classes as one-hot rows
	for i in range(batch_size):
		# Randomly select one of the three classes for the sample in this batch
		temp_class = classes[rd.randint(0, output_classes-1),:]
		# Repeat until proper label is found
		proper_label = False
		while proper_label == False:
			# Generate random integer
			temp_index = rd.randint(0, f_length-num_steps-1)
			# Read data from CSV and write as matrix
			temp = pd.read_csv(filename, skiprows=temp_index, nrows=num_steps, header=None)
			temp = temp.as_matrix()
			# If the label is correct...
			if np.array_equal(temp[num_steps-1, input_features+1:input_features+1+output_classes], temp_class):
				# Return features in specified columns
				feature_batch[i,:,:] = temp[:,1:input_features+1]
				# Return *last label* in specified columns
				label_batch[i,:] = temp[num_steps-1, input_features+1:input_features+1+output_classes]
				# Break from while loop and increment class
				proper_label = True

	# Return feature and label batches
	return feature_batch, label_batch

def extract_data_window(filename, batch_size, num_steps, input_features, output_classes, step):
	"""
	Extract features and labels from filename.csv in rolling-window batches
	Returns:
	feature_batch ~ [batch_size, num_steps, input_features]
	label_batch ~ [batch_size, output_classes]
	"""

	# Initialize numpy arrays for return value placeholders
	feature_batch = np.zeros((batch_size,num_steps,input_features))
	label_batch = np.zeros((batch_size, output_classes))

	# Import data from CSV as a sliding window:
	# First, import data starting from t = minibatch to t = minibatch + num_steps
	# ... add feature data to feature_batch[0, :, :]
	# ... add label data to label_batch[batch, :, :]
	# Then, import data starting from t = minibatch + 1 to t = minibatch + num_steps + 1
	# ... add feature data to feature_batch[1, :, :]
	# ... add label data to label_batch[batch, :]
	# Repeat for all batches.
	for i in range(batch_size):
		temp = pd.read_csv(filename, skiprows=minibatch*batch_size+i, nrows=num_steps, header=None)
		temp = temp.as_matrix()
		# Return features in specified columns
		feature_batch[i,:,:] = temp[:,1:input_features+1]
		# Return *last label* in specified columns
		label_batch[i,:] = temp[num_steps-1, input_features+1:input_features+1+output_classes]

	# Return feature and label batches
	return feature_batch, label_batch


# ----------------------------------------------------
# Importing FoG Dataset Batches
# ----------------------------------------------------
# Create placeholders for inputs and target values
# Input dimensions: BATCH_SIZE x NUM_STEPS x INPUT_FEATURES
# Target dimensions: BATCH_SIZE x OUTPUT_CLASSES
inputs = tf.placeholder(tf.float32, [lstm.batch_size, lstm.num_steps, lstm.input_features], name="Input_Placeholder")
targets = tf.placeholder(tf.float32, [lstm.batch_size, lstm.output_classes], name="Target_Placeholder")


# ----------------------------------------------------
# Building a Multilayer LSTM
# ----------------------------------------------------
# Build LSTM cells
with tf.name_scope("LSTM_Network"):
	cells = []
	for _ in range(lstm.num_lstm_layers):
		# Creating basic LSTM cell
		cell = tf.contrib.rnn.BasicLSTMCell(lstm.num_lstm_hidden)
		# Adding dropout wrapper to cell
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=I_KEEP_PROB, output_keep_prob=O_KEEP_PROB)
		# Stacking LSTM cells
		cells.append(cell)
	stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

# Initialize weights and biases for fully-connected layer.
with tf.name_scope("FCN_Variables"):
	W = init_values([lstm.num_lstm_hidden, lstm.output_classes])
	tf.summary.histogram('Weights',W)
	b = init_values([lstm.output_classes])
	tf.summary.histogram('Biases',b)

# Add LSTM cells to dynamic_rnn and implement truncated BPTT
initial_state = state = stacked_lstm.zero_state(lstm.batch_size, tf.float32)
logits = []
with tf.name_scope("Dynamic_RNN"):
	for i in range(lstm.num_steps):
		# Obtain output at each step
		output, state = tf.nn.dynamic_rnn(stacked_lstm, inputs[:,i:i+1,:], initial_state=state)
# Obtain final output and convert to logit
# Reshape output to remove extra dimension
output = tf.reshape(output,[lstm.batch_size,lstm.num_lstm_hidden])
with tf.name_scope("Append_Output"):
	# Obtain logits by passing output
	for j in range(lstm.batch_size):
		logit = tf.matmul(output[j:j+1,:], W) + b
		logits.append(logit)
# Converting logits array to tensor and reshaping the array such that it has the arrangement:
# [class1 class2 ...] batch = 1
# [class1 class2 ...] batch = 2 ...
with tf.name_scope("Reformat_Logits"):
	logits = tf.reshape(tf.convert_to_tensor(logits), [lstm.batch_size, lstm.output_classes])


# ----------------------------------------------------
# Calculate Loss and Define Optimizer
# ----------------------------------------------------
# Calculating softmax cross entropy of labels and logits
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits)
loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Obtain predictions from logits
predictions = tf.nn.softmax(logits)


# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver()	# Instantiate Saver class
with tf.Session() as sess:
	# Create Tensorboard graph
	writer = tf.summary.FileWriter(filewriter_path, sess.graph)
	merged = tf.summary.merge_all()

	# If there is a model checkpoint saved, load the checkpoint. Else, initialize variables.
	if LOAD_FILE:
		# Restore saved session
		saver.restore(sess, save_path)
	else:
		# Initialize the variables
		sess.run(init)

	# Training the network

	# Determine whether to use sliding-window or minibatching
	if MINI_BATCH:
		step_range = NUM_TRAINING 		# Set step range for training
		v_step_range = NUM_VALIDATION	# Set step range for validation
		window_int_t = 1	# Select window interval for training
		window_int_v = 1	# Select window interval for validation
	else: # If sliding window...
		step_range = file_length 		# Set step range for training
		v_step_range = v_file_length 	# Set step range for validation
		window_int_t = WINDOW_INT_t 	# Select window interval for training
		window_int_v = WINDOW_INT_v 	# Select window interval for validation

	# Obtain start time
	start_time = time.time()
	# Initialize optimal loss
	loss_op = 0

	# Set number of trials to NUM_TRAINING
	for step in range(0,step_range,window_int_t):

		# Initialize optimal model saver to False
		save_op = False

		try:	# While there is no out-of-bounds exception...

			# Obtaining batch of features and labels from TRAINING dataset(s)
			if MINI_BATCH: # Balanced minibatches
				features, labels = extract_data_balanced(tDataset, lstm.batch_size, lstm.num_steps, lstm.input_features, lstm.output_classes, file_length)
			else: # Rolling window
				features, labels = extract_data_window(tDataset, lstm.batch_size, lstm.num_steps, lstm.input_features, lstm.output_classes, step)

		except:
			break

		# Set optional conditional for network training
		if True:
			# Print step
			print("\nOptimizing at step", step)
			# Input data
			data = {inputs: features, targets:labels}
			# Run optimizer, loss, and predicted error ops in graph
			predictions_, targets_, _, loss_ = sess.run([predictions, targets, optimizer, loss], feed_dict=data)

			# Evaluate network and print data in terminal periodically
			with tf.name_scope("Validation"):
				# Conditional statement for validation and printing
				if step % 100 == 0:
					print("\nMinibatch train loss at step", step, ":", loss_)

					# Evaluate network
					test_loss = []
					for step_num in range(0,v_step_range,window_int_v):

						try:	# While there is no out-of-bounds exception...

							# Obtaining batch of features and labels from VALIDATION dataset(s)
							if MINI_BATCH:  # Balanced minibatches
								v_features, v_labels =  extract_data_balanced(vDataset, lstm.batch_size, lstm.num_steps, lstm.input_features, lstm.output_classes, v_file_length)
							else: # Rolling window
								v_features, v_labels =  extract_data_window(vDataset, lstm.batch_size, lstm.num_steps, lstm.input_features, lstm.output_classes, step_num)

						except:
							break
						
						# Input data and run session to find loss
						data_test = {inputs: v_features, targets: v_labels}
						loss_test = sess.run(loss, feed_dict=data_test)
						test_loss.append(loss_test)

					# Print test loss
					print("Test loss: %.3f" % np.mean(test_loss))
					# For the first step, set optimal loss to test loss
					if step == 0:
						loss_op = np.mean(test_loss)
					# If test_loss < optimal loss, overwrite optimal loss
					if np.mean(test_loss) < loss_op:
						loss_op = np.mean(test_loss)
						save_op = True 	# Save model as new optimal model

					# Print predictiond and targets for reference
					print("Predictions:")
					print(predictions_)
					print("Targets:")
					print(targets_)

			# Save and overwrite the session at each training step
			saver.save(sess, save_path)
			# Save the model if loss over the test set is optimal
			if save_op:
				saver.save(sess,save_path_op)

			# Writing summaries to Tensorboard at each training step
			summ = sess.run(merged)
			writer.add_summary(summ,step)

		# Conditional statement for calculating time remaining and percent completion
		if step % 10 == 0:

			# Report percent completion
			if MINI_BATCH: # Balanced minibatches
				p_completion = 100*step/NUM_TRAINING
			else: # Rolling window
				p_completion = 100*step/file_length
			print("\nPercent completion: %.3f%%" % p_completion)

			# Print time remaining
			avg_elapsed_time = (time.time() - start_time)/(step+1)
			if MINI_BATCH: # Balanced minibatches
				sec_remaining = avg_elapsed_time*(NUM_TRAINING-step)
			else: # Sliding window
				sec_remaining = round(avg_elapsed_time*(file_length-step)/window_int_t)
			min_remaining = round(sec_remaining/60)
			print("\nTime Remaining: %d minutes" % min_remaining)

	# Close the writer
	writer.close()