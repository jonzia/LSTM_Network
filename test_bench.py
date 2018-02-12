# ----------------------------------------------------
# LSTM Network Test Bench for LSTM_Network v1.3.2
# Created by: Jonathan Zia
# Last Modified: Monday, Feb 12, 2018
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
BATCH_SIZE = 1      # Batch size
NUM_STEPS = 100     # Max steps for BPTT
NUM_LSTM_LAYERS = 1 # Number of LSTM layers
NUM_LSTM_HIDDEN = 5 # Number of LSTM hidden units
OUTPUT_CLASSES = 3  # Number of classes / FCL output units
INPUT_FEATURES = 9  # Number of input features
I_KEEP_PROB = 1.0   # Input keep probability / LSTM cell
O_KEEP_PROB = 1.0   # Output keep probability / LSTM cell
WINDOW_INT = 10     # Rolling window step interval

# ----------------------------------------------------
# Input data files
# ----------------------------------------------------
# Specify filenames
# Root directory:
dir_name = "/Users/username/Documents"
with tf.name_scope("Training_Data"):    # Testing dataset
    Dataset = os.path.join(dir_name, "/dataset/filename.csv")
with tf.name_scope("Model_Data"):       # Model load path
    load_path = os.path.join(dir_name, "checkpoints/model")
with tf.name_scope("Filewriter_Data"):  # Filewriter save path
    filewriter_path = os.path.join(dir_name, "output")
with tf.name_scope("Output_Data"):      # Output data filenames (.txt)
    # These .txt files will contain prediction and target data respectively for Matlab analysis
    prediction_file = os.path.join(dir_name, "predictions.txt")
    target_file = os.path.join(dir_name, "targets.txt")

# Obtain length of testing and validation datasets
file_length = len(pd.read_csv(Dataset))

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

def extract_data(filename, batch_size, num_steps, input_features, output_classes, minibatch):
    """
    Extract features and labels from filename.csv in rolling-window batches
    Returns:
    feature_batch ~ [batch_size, num_steps, input_features]
    label_batch ~ [batch_size, num_steps]
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
    b = init_values([OUTPUT_CLASSES])

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
# Calculate Loss
# ----------------------------------------------------
<<<<<<< HEAD
# Calculating softmax cross entropy of labels and logits
=======
# Calculating sigmoid cross entropy of labels and logits
>>>>>>> origin/master
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits)
loss = tf.reduce_mean(loss)

# Obtain predictions from logits
predictions = tf.nn.softmax(logits)

# Obtain histogram data
tf.summary.histogram('Predictions', predictions)
tf.summary.histogram('Targets', targets)

# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
saver = tf.train.Saver()    # Instantiate Saver class
with tf.Session() as sess:
    # Create Tensorboard graph
    writer = tf.summary.FileWriter(filewriter_path, sess.graph)
    merged = tf.summary.merge_all()

    # Restore saved session
    saver.restore(sess, load_path)

    # Running the network
    # Set range (prevent index out-of-range exception for rolling window)
    for step in range(0,file_length,WINDOW_INT):

        try: # While there is no out-of-bounds exception
            # Obtaining batch of features and labels from TRAINING dataset(s)
            features, labels = extract_data(Dataset, BATCH_SIZE, NUM_STEPS, INPUT_FEATURES, OUTPUT_CLASSES,step)
        except:
            break

        # Input data
        data = {inputs: features, targets:labels}
        # Run and evalueate for summary variables, loss, predictions, and targets
        summary, loss_, pred, tar = sess.run([merged, loss, predictions, targets], feed_dict=data)

        # Report parameters
        if True:    # Conditional statement for filtering outputs
            p_completion = math.floor(100*step/file_length)
            print("\nLoss: %.3f, Percent Completion: " % loss_, p_completion)
            print("\nPredictions:")
            print(pred)
            print("\nTargets:")
            print(tar)

            # Write results to file for Matlab analysis
            # Write predictions
            with open(prediction_file, 'a') as file_object:
                np.savetxt(file_object, pred)
            # Write targets
            with open(target_file, 'a') as file_object:
                np.savetxt(file_object, tar)

        # Writing summaries to Tensorboard
        writer.add_summary(summary,step)

    # Close the writer
    writer.close()