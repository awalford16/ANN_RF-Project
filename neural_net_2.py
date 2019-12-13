# An improved approach to the original neural network
import numpy as np
import math
import random

class NeuralNet():
    def __init__(self, data, inputs, hidden, outputs, lr):
        self.x = data
        self.input_nodes = inputs
        self.hidden_nodes = hidden
        self.output_nodes = outputs
        self.learning_rate = lr

        # Initialise neruon values
        self.values_h1 = np.zeros(hidden)
        self.values_h2 = np.zeros(hidden)
        self.values_o = np.zeros(outputs)

        # Initialise the weights between -0.5 and 0.5
        np.random.seed(1000)
        self.weights_ih = np.random.rand(hidden, inputs) - 0.5
        self.weights_hh = np.random.rand(hidden, hidden) - 0.5
        self.weights_ho = np.random.rand(outputs, hidden) - 0.5

        # Initialise the biases
        self.bias_h1 = np.zeros((1, hidden))
        self.bias_h2 = np.zeros((1, hidden))
        self.bias_o = np.zeros((1, outputs))

    # Step forward and back propagation processing
    def feed_forward(self, data):
        # Process first hidden layer
        h1 = np.dot(data, self.weights_ih.T) + self.bias_h1
        h1_sig = self.sigmoid(h1)
        print(h1_sig.mean())
        self.values_h1 = h1_sig

        # Process 2nd hidden layer
        h2 = np.dot(h1_sig, self.weights_hh.T) + self.bias_h2
        h2_sig = self.sigmoid(h2)
        print(h2_sig.mean())
        self.values_h2 = h2_sig

        # Process output layer
        output_layer = np.dot(h2_sig, self.weights_ho.T) + self.bias_o
        output = self.sigmoid(output_layer)
        print(output.mean())
        self.values_o = output

    # Back propagation
    def back_prop(self, data, actual):
        # Get the difference between predicted and actual value
        loss = np.subtract(self.values_o, actual)

        # Get squared error
        error_sum = np.power(loss, 2).sum()

        # Determine derivative of output
        error_der = self.sigmoid_der(self.values_o) 

        # Determine gradient of output
        output_gradient = np.dot(loss, error_der)
        
        # Update the weights and biases between hidden layer 2 and output layer
        self.weights_ho += self.update_weights(self.values_h2, output_gradient)
        self.bias_o += self.update_biases(output_gradient)

        hidden2_error_der = self.sigmoid_der(self.values_h2) 
        hidden2_gradient = np.dot(output_gradient, self.weights_ho) * hidden2_error_der

        # Update between two hidden nodes
        self.weights_hh += self.update_weights(self.values_h1, hidden2_gradient)
        self.bias_h2 += self.update_biases(hidden2_gradient)

        # Determine gradient of hidden layer 1
        hidden1_error_der = self.sigmoid_der(self.values_h1)
        hidden1_gradient = np.dot(hidden2_gradient, self.weights_hh.T) * hidden1_error_der

        # Update weights and biases between inputs and first hidden layer
        self.weights_ih += self.update_weights(data, hidden1_gradient)
        self.bias_h1 += self.update_biases(hidden1_gradient)

        print(self.weights_hh)
        
        return error_sum

    # Update biases
    def update_biases(self, gradient):
        return self.learning_rate * np.sum(gradient, axis=0)

    # Update the weights
    def update_weights(self, inputs, gradient):
        return (self.learning_rate * np.dot(inputs.T, gradient)).T

    # Get the derivative of the sigmoid function
    def sigmoid_der(self, x):
        return x * (1 - x)

    # Sigmoid function to act as activiation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_accuracy(self, error_sum, actual):
        # If the RMSE is more than .5 away, respond incorrectly classified
        return error_sum / len(actual)
