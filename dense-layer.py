# Dependencies
import numpy as np
import nnfs
from nnfs.datasets import spiral_data # A library by Sentdex, which provides different datasets to work with

nnfs.init() # to ensure a constant random state, sets seed to zero (default)

# Implementing a Class for a layer, such that it can be used to create several layer objects:
class dense_layer:
	
	# initializing the layer 
	def __init__(self, n_inputs, n_neurons):
		# Randomly initializing weights
		self.weights = np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons)) # Initializing zero bias for each neuron in this layer

	# forward pass
	def forward_pass(self, inputs):
		# dot-product of inputs and weights and adding biases
		self.output = np.dot(inputs, self.weights) + self.biases

# Dataset
X, y = spiral_data(samples=100, classes=3)

# Layer-1:
layer1 = dense_layer(2,3)

# forward pass on layer-1:
layer1.forward_pass(X)

# output of the forward pass:
print(layer1.output)