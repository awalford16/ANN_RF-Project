import numpy as np
import math

# An improved approach to the original neural network
class NeuralNet:
    def __init__(self, inputs, hidden, layers, outputs):
        self.input_nodes = inputs
        self.hidden_nodes = hidden
        self.layers = layers
        self.output_nodes = outputs

        # Create array for the data values
        self.data = np.zeros(inputs)

        # Randomise weight matrix for each neuron connection
        self.weights_ih = np.random.random((self.hidden_nodes, self.input_nodes))
        self.weights_hh = np.random.random((self.hidden_nodes, self.hidden_nodes))
        self.weights_ho = np.random.random((self.output_nodes, self.hidden_nodes))

        # Carete random arrays for biases
        self.bias_h1 = np.random.random((self.hidden_nodes, 1))
        self.bias_h2 = np.random.random((self.hidden_nodes, 1))
        self.bias_o = np.random.random((self.output_nodes, 1))

    # set the data for the input values
    def set_input_data(self, data):
        # Foreach data feature, assign it to the corresponding value index
        for i in range(data.shape[0]):
            self.data[i] = data[i]

    # Calculate the outputs of each hidden layer
    def compute_layer(self, data, weights, bias):
        # Compute first layer and add bias
        out = weights.dot(data)
        out = np.add(out, bias)

        # Calculate sigmoid of first hidden layer
        return self.sigmoid(out)


    def feed_forward(self, data):
        self.set_input_data(data)

        # Initialise values for first layer computation
        values = self.data
        weights = self.weights_ih
        bias = self.bias_h1
        
        # Loop through each of the hidden layers
        for i in range(self.layers):
            if i != 0:
                weights = self.weights_hh
                bias = self.bias_h2

            values = self.compute_layer(values, weights, bias)

        # compute output
        output = self.compute_layer(values, self.weights_ho, self.bias_o)
        print(output)


    # Sigmoid function to act as activiation function
    def sigmoid(self, x):
        for i, array in enumerate(x):
            for j, val in enumerate(array):
                x[i][j] = 1 / (1 + math.exp(-val))

        return x