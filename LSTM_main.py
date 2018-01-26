# ----------------------------------------------------
# LSTM Network Implementation using Tensorflow 1.1.0
# Created by: Jonathan Zia
# Last Modified: Friday, Jan 26, 2018
# Georgia Institute of Technology
# ----------------------------------------------------
import tensorflow as tf
import numpy as np
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
# User-Defined Methods
# ----------------------------------------------------
def get_len(filename):
	"""
	Obtain CSV File Length
	Returns: Number of lines in CSV file "filename"
	"""
	with open(filename) as csvfile:
		temp = csv.DictReader(csvfile)
		return len(list(temp))

def init_values(shape):
	"""
	Initialize Weight and Bias Matrices
	Returns: Tensor of shape "shape" w/ normally-distributed values
	"""
	temp = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(temp)

def extract_data(file_list, batch_size, num_steps):
	"""
	Extract features and labels from list of data files
	Returns: Feature and label batches of size BATCH_SIZE
		and total number of input examples across all input
		files in "file_list"
	"""
	# Constructing a queue of filenames for training
	filename = tf.train.string_input_producer(file_list)

	# Get file length
	file_length = 0
	for i in range(len(file_list)):
		file_length = file_length + get_len(file_list[i])

	# Reading one line of the CSV input file
	reader = tf.TextLineReader(name='Text_Line_Reader')
	key, value = reader.read(filename, name='Reader')

	# Setting default values in case of empty columns
	defaults = [[0.0] for _ in range(INPUT_FEATURES+2)]
	# Storing comma-delimited data from "value" into individual variables
	decoded_data = tf.decode_csv(
		value, record_defaults = defaults, name='Decode_CSV')

	# Splitting data into features and labels
	# Data is formatted as follows:
	# feature_batch:	[ [time1]
	#					  [time2] ] (batch1)

	#					[ [time1]
	#					  [time2] ] (batch2)
	#
	# label_batch:		[time1 time2] (batch1)
	#					[time1 time2] (batch2)

	features = tf.reshape(tf.stack(decoded_data[1:INPUT_FEATURES+1]),[1,INPUT_FEATURES])
	labels = tf.reshape(decoded_data[INPUT_FEATURES+1],[1,1])
	feature_batch, label_batch = tf.train.batch([features, labels], 
		batch_size=num_steps)
	feature_batch, label_batch = tf.train.batch([feature_batch, label_batch], batch_size=batch_size)
	label_batch = tf.reshape(label_batch, [batch_size,num_steps])
	feature_batch = tf.reshape(feature_batch, [batch_size,num_steps,INPUT_FEATURES])


	# Return labels, features, and file length
	return feature_batch, label_batch, file_length


# ----------------------------------------------------
# Importing FoG Datasets
# ----------------------------------------------------
# Obtaining batch of features and labels from TRAINING dataset(s)
with tf.name_scope("Training_Data"):
	tDataset = [""]
	features, labels, file_length = extract_data(tDataset, BATCH_SIZE, NUM_STEPS)

# Obtaining batch of features and labels from VALIDATION dataset(s)
with tf.name_scope("Validation_Data"):
	vDataset = [""]
	v_features, v_labels, v_file_length = extract_data(vDataset, BATCH_SIZE, NUM_STEPS)


# ----------------------------------------------------
# Building a Multilayer LSTM
# ----------------------------------------------------
# Create placeholders for inputs and target values
# Input dimensions: BATCH_SIZE x NUM__STEPS x 9
# Target dimensions: BATCH_SIZE x NUM_STEPS (x 1)
inputs = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, INPUT_FEATURES], name="Input_Placeholder")
targets = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS], name="Target_Placeholder")

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
for i in range(NUM_STEPS):
	# Obtain output at each step
	output, state = tf.nn.dynamic_rnn(stacked_lstm, inputs[:,i:i+1,:], initial_state=state)
	# Reshape output to remove extra dimension
	output = tf.reshape(output,[BATCH_SIZE,NUM_LSTM_HIDDEN])

	# Obtain logits by passing output
	for j in range(BATCH_SIZE):
		logit = tf.nn.sigmoid(tf.matmul(output[j:j+1,:], W) + b)
		logits.append(logit)
# Converting logits array to tensor and reshaping the array such that it has the arrangement:
# [time1 time2] batch = 1
# [time1 time2] batch = 2 ...
logits = tf.reshape(tf.convert_to_tensor(logits), [BATCH_SIZE, NUM_STEPS])

# ----------------------------------------------------
# Calculate Loss and Define Optimizer
# ----------------------------------------------------
# Calculating mean squared error of labels and logits
loss = tf.losses.mean_squared_error(labels, logits)
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
	for step in range(int(file_length/BATCH_SIZE)):
		# Input data
		data = {inputs: features.eval(), targets:labels.eval()}
		# Run optimizer, loss, and predicted error ops in graph
		_, loss_train = sess.run([optimizer,loss], feed_dict=data)

		# Evaluate network and print data in terminal periodically
		if step % 50 == 0:
			print("Minibatch train loss at step", step, ":", loss_train)

			# Evaluate network
			test_loss = []
			for batch_num in range(int(v_file_length/BATCH_SIZE)):
				data_test = {inputs: v_features.eval(), targets: v_labels.eval()}
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