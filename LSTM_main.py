# ----------------------------------------------------
# LSTM Network Implementation using Tensorflow 1.2.3
# Created by: Jonathan Zia
# Last Modified: Wednesday, Feb 7, 2018
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
BATCH_SIZE = 3      # Batch size
NUM_STEPS = 100     # Max steps for BPTT
NUM_LSTM_LAYERS = 1 # Number of LSTM layers
NUM_LSTM_HIDDEN = 5 # Number of LSTM hidden units
OUTPUT_CLASSES = 3  # Number of classes / FCL output units
INPUT_FEATURES = 9  # Number of input features
I_KEEP_PROB = 1.0   # Input keep probability / LSTM cell
O_KEEP_PROB = 1.0   # Output keep probability / LSTM cell
WINDOW_INT_t = 1    # Rolling window step interval for training
WINDOW_INT_v = 10   # Rolling window step interval for validation

# ----------------------------------------------------
# Input data files
# ----------------------------------------------------
# Specify filenames
with tf.name_scope("Training_Data"):    # Training dataset
    tDataset = ""
with tf.name_scope("Validation_Data"):  # Validation dataset
    vDataset = ""
with tf.name_scope("Model_Data"):       # Model save path
    save_path = "/checkpoints/model"
with tf.name_scope("Filewriter_Data"):  # Filewriter save path
    filewriter_path = "/output"

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

def extract_data(filename, batch_size, num_steps, input_features, output_classes, epoch):
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
    # First, import data starting from t = epoch to t = epoch + num_steps
    # ... add feature data to feature_batch[0, :, :]
    # ... add label data to label_batch[batch, :, :]
    # Then, import data starting from t = epoch + 1 to t = epoch + num_steps + 1
    # ... add feature data to feature_batch[1, :, :]
    # ... add label data to label_batch[batch, :]
    # Repeat for all batches.
    for i in range(batch_size):
        temp = pd.read_csv(filename, skiprows=epoch*batch_size+i, nrows=num_steps, header=None)
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
inputs = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, INPUT_FEATURES], name="Input_Placeholder")
targets = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_CLASSES], name="Target_Placeholder")


# ----------------------------------------------------
# Building a Multilayer LSTM
# ----------------------------------------------------
# Build LSTM cells
with tf.name_scope("LSTM_Network"):
    cells = []
    for _ in range(NUM_LSTM_LAYERS):
        # Creating basic LSTM cell
        cell = tf.contrib.rnn.BasicLSTMCell(NUM_LSTM_HIDDEN)
        # Adding dropout wrapper to cell
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=I_KEEP_PROB, output_keep_prob=O_KEEP_PROB)
        # Stacking LSTM cells
        cells.append(cell)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

# Initialize weights and biases for fully-connected layer.
with tf.name_scope("FCN_Variables"):
    W = init_values([NUM_LSTM_HIDDEN, OUTPUT_CLASSES])
    tf.summary.histogram('Weights',W)
    b = init_values([OUTPUT_CLASSES])
    tf.summary.histogram('Biases',b)

# Add LSTM cells to dynamic_rnn and implement truncated BPTT
initial_state = state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
logits = []
with tf.name_scope("Dynamic_RNN"):
    for i in range(NUM_STEPS):
        # Obtain output at each step
        output, state = tf.nn.dynamic_rnn(stacked_lstm, inputs[:,i:i+1,:], initial_state=state)
# Obtain final output and convert to logit
# Reshape output to remove extra dimension
output = tf.reshape(output,[BATCH_SIZE,NUM_LSTM_HIDDEN])
with tf.name_scope("Append_Output"):
    # Obtain logits by passing output
    for j in range(BATCH_SIZE):
        logit = tf.matmul(output[j:j+1,:], W) + b
        logits.append(logit)
# Converting logits array to tensor and reshaping the array such that it has the arrangement:
# [class1 class2 ...] batch = 1
# [class1 class2 ...] batch = 2 ...
with tf.name_scope("Reformat_Logits"):
    logits = tf.reshape(tf.convert_to_tensor(logits), [BATCH_SIZE, OUTPUT_CLASSES])

# ----------------------------------------------------
# Calculate Loss and Define Optimizer
# ----------------------------------------------------
# Calculating softmax cross entropy of labels and logits
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits)
loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Obtain predictions from logits
predictions = tf.softmax(logits)

# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver()    # Instantiate Saver class
with tf.Session() as sess:
    # Create Tensorboard graph
    writer = tf.summary.FileWriter(filewriter_path, sess.graph)
    merged = tf.summary.merge_all()
    # Initialize the variables
    sess.run(init)

    # Training the network
    # Set range (prevent index out-of-range exception for rolling window)
    rng = math.floor((file_length - BATCH_SIZE - NUM_STEPS + 2) / BATCH_SIZE)
    # Set rolling window step between batches to WINDOW_INT
    for step in range(0,rng,WINDOW_INT_t):
        # Obtaining batch of features and labels from TRAINING dataset(s)
        features, labels = extract_data(tDataset, BATCH_SIZE, NUM_STEPS, INPUT_FEATURES, OUTPUT_CLASSES, step)

        # Set conditional statement for performing optimization/analysis
        if True:
            # Input data
            data = {inputs: features, targets:labels}
            # Run optimizer, loss, and predicted error ops in graph
            predictions_, targets_, _, loss_ = sess.run([predictions, targets, optimizer, loss], feed_dict=data)

            # Evaluate network and print data in terminal periodically
            with tf.name_scope("Validation"):
                # Set conditional statement for validation and printing
                if step % 50 == 0:
                    print("\nMinibatch train loss at step", step, ":", loss_)

                    # Evaluate network
                    test_loss = []
                    # Set range (prevent index out-of-range exception for rolling window)
                    v_rng = math.floor((v_file_length - BATCH_SIZE - NUM_STEPS + 2) / BATCH_SIZE)
                    for step_num in range(0,v_rng,WINDOW_INT_v):

                        # Obtaining batch of features and labels from VALIDATION dataset(s)
                        v_features, v_labels =  extract_data(vDataset, BATCH_SIZE, NUM_STEPS, INPUT_FEATURES, OUTPUT_CLASSES, step_num)

                        # Input data and run session to find loss
                        data_test = {inputs: v_features, targets: v_labels}
                        loss_test = sess.run(loss, feed_dict=data_test)
                        test_loss.append(loss_test)
                    print("Test loss: %.3f" % np.mean(test_loss))

                    # Report percent completion
                    p_completion = 100*step/rng
                    print("Percent completion: %.3f%%\n" % p_completion)

                    # Print predictiond and targets for reference
                    print("Predictions:")
                    print(predictions_)
                    print("Targets:")
                    print(targets_)

            # Save and overwrite the session
            saver.save(sess, save_path)

            # Writing summaries to Tensorboard
            summ = sess.run(merged)
            writer.add_summary(summ,step)

    # Close the writer
    writer.close()
