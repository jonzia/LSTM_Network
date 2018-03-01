# ----------------------------------------------------
# Network Class for LSTM Network 1.3.5
# Created by: Jonathan Zia
# Last Modified: Monday, Feb 26, 2018
# Georgia Institute of Technology
# ----------------------------------------------------

class Network():
	"""Class containing all network parameters for use
	across LSTM_main and test_bench programs."""

	def __init__(self):
		"""Initialize network attributes"""
		
		# Architecture
		self.batch_size = 50		# Batch size
		self.num_steps = 100		# Max steps for BPTT
		self.num_lstm_layers = 2	# Number of LSTM layers
		self.num_lstm_hidden = 15	# Number of LSTM hidden units
		self.output_classes = 3		# Number of classes / FCL output units
		self.input_features = 9		# Number of input features
		self.i_keep_prob = 0.9		# Input keep probability / LSTM cell
		self.o_keep_prob = 0.9		# Output keep probability / LSTM cell


		# Decay type can be 'none', 'exp', 'inv_time', or 'nat_exp'
		self.decay_type = 'exp'		# Set decay type for learning rate
		self.learning_rate_init = 0.001		# Set initial learning rate for optimizer (default 0.001) (fixed LR for 'none')
		self.learning_rate_end = 0.00001	# Set ending learning rate for optimizer