# Dependencies
import numpy as np
import nnfs
from nnfs.datasets import spiral_data # A library by Sentdex, which provides different datasets to work with

nnfs.init() # to ensure a constant random state, sets seed to zero (default)

# Implementing a Class for a layer, such that it can be used to create several layer objects:
class DenseLayer:
	
	# initializing the layer 
	def __init__(self, n_inputs, n_neurons):
		# Randomly initializing weights
		self.weights = np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons)) # Initializing zero bias for each neuron in this layer

	# forward pass
	def forward_pass(self, inputs):
		# dot-product of inputs and weights and adding biases
		self.output = np.dot(inputs, self.weights) + self.biases

# Implementing ReLU Activation function to be used in dense-layer
class ReLU:

	# Forward pass method to accept inputs from the prev layer and apply ReLU
	def forward_pass(self, inputs):
		self.output = np.maximum(0, inputs)

# Implementing Softmax Activation function for the final layer, to get probabilities of the decision:
class SoftMax:

	# Forward pass method to accept inputs from the final hidden layer, and normalize them:
	def forward_pass(self, inputs):
		# unnormalized probabilites (numerator of the softmax function)
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims= True))
		# normalizing each of the individula value in the numerator:
		probabilities = exp_values/np.sum(exp_values, axis= 1, keepdims= True)
		# output probabilities/decisions:
		self.output = probabilities



# Dataset
X, y = spiral_data(samples=100, classes=3)

# Layer-1: 2 inputs with 3 neuron, outputs 3 values
layer1 = DenseLayer(2,3)
#activation-1:
activation1 = ReLU()
# layer-2, 3 inputs (accepting the output of layer-1), 3 neurons
layer2 = DenseLayer(3, 3)
# activation-2: Softmax, assigning layer-2 as final hidden layer:
activation2 = SoftMax()

#forward pass through layer1:
layer1.forward_pass(X)
activation1.forward_pass(layer1.output)
layer2.forward_pass(activation1.output)
activation2.forward_pass(layer2.output)

print(activation2.output[:5])