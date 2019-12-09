import numpy as np
import math
import random

class NeuralNet:
    def __init__(self, inputs, hidden, layers, outputs, rate):
        # initialise all passed in values
        self.input_nodes = inputs
        self.hidden_nodes = hidden

        # Allow for multiple hidden layers
        self.hidden_layers = layers
        self.output_nodes = outputs

        # Total amount of neurons in network
        self.total = inputs + (hidden * layers) + outputs
        self.learning_rate = rate
        self.values = np.zeros(self.total)

        # Get index for output layer inputs
        self.output_inputs = inputs + (hidden * (layers - 1))

        # Create a matrix to contain random weights based on amount of nodes
        self.weights = np.zeros((self.total, self.total))
        self.bias = np.zeros(self.total)
        
        #random.seed(10000)
        # Set radomised intial values for bias and weights
        for i in range(self.input_nodes, self.total):
            self.bias[i] = random.random() / random.random()
            for j in range(i + 1, self.total):
                self.weights[i][j] = random.random() - 1

    # Functionality within hidden layer
    def hidden_layer(self, input_min, input_max):
        # Loop through each of the hidden nodes for that layer
        for i in range(input_max, input_max + self.hidden_nodes):
            weight = 0.0

            # For each of the features, multiply by weight
            for j in range(input_min, input_max):
                weight += self.weights[j][i] * self.values[j]
            
            weight += self.bias[i]
            #print(self.sigmoid(weight))
            # Sigmoid function as non-linear activation function
            self.values[i] = self.sigmoid(weight)

    # Get results of output layer
    def output_layer(self, input_min, input_max):
        # Loop through each of the output nodes
        for i in range(input_max, self.total):
            # sum weighted hidden nodes for each output node, compare threshold, apply sigmoid
            weight = 0.0

            # Loop through hidden nodes connected to output layer and update weights
            for j in range(input_min, input_max):
                # Randomised weight multiplied by data values
                weight += self.weights[j][i] * self.values[j]

            weight += self.bias[i]

            # logistic function
            self.values[i] = self.sigmoid(weight)

    # set the data for the input values
    def set_input_data(self, data):
        # Foreach data feature, assign it to the corresponding value index
        for i in range(data.shape[0]):
            self.values[i] = data[i]

    # Run neural net process
    def feed_forward(self, data):
        # Set values based on passed in data
        self.set_input_data(data)

        # Loop through each hidden layer
        for i in range(self.hidden_layers):
            # If first hidden layer, input indexes are frominput layer
            if i == 0:
                self.hidden_layer(0, self.input_nodes)
                continue
            
            # else input indexes are from hidden layer interation
            self.hidden_layer(self.input_nodes + (self.hidden_nodes * (i - 1)), self.input_nodes + (self.hidden_nodes * i))

        # Get values for output layer
        self.output_layer(self.output_inputs, self.output_inputs + self.hidden_nodes)

    # Calculate error of prediction
    def back_propagation(self, expected):
        squared_error = 0.0

        # Loop through output nodes
        for i in range(self.output_inputs + self.hidden_nodes, self.total):
            # find difference in expected and predicted
            error = expected - self.values[i]

            # Square the error and set gradient based on error
            squared_error = math.pow(error, 2)
            error_gradient = self.values[i] * (1 - self.values[i]) * error

            self.update_weights(error_gradient, i)

        return squared_error

    # Update weights to improve model accuracy
    def update_weights(self, gradient, output_node):
        for i in range(self.input_nodes, (self.hidden_nodes * self.hidden_layers)):
            delta = self.learning_rate * self.values[i] * gradient
            self.weights[i][output_node] += delta
            hidden_gradient = self.values[i] * (1 - self.values[i]) * gradient * self.weights[i][output_node]

            # Update input nodes to hidden nodes
            for k in range(self.input_nodes):
                delta = self.learning_rate * self.values[k] * hidden_gradient
                self.weights[k][i] += delta

    # Sigmoid function to act as activiation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
