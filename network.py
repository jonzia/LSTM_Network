# ----------------------------------------------------
# Network Class for LSTM Network 1.3.3
# Created by: Jonathan Zia
# Last Modified: Wednesday, Feb 14, 2018
# Georgia Institute of Technology
# ----------------------------------------------------

class Network():
	"""Class containing all network parameters for use
	across LSTM_main and test_bench programs."""

	def __init__(self):
		"""Initialize network attributes"""
		
		# Architecture
		self.batch_size = 9			# Batch size
		self.num_steps = 100		# Max steps for BPTT
		self.num_lstm_layers = 1	# Number of LSTM layers
		self.num_lstm_hidden = 10	# Number of LSTM hidden units
		self.output_classes = 3		# Number of classes / FCL output units
		self.input_features = 9		# Number of input features