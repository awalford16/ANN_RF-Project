# An improved approach to the original neural network
import numpy as np
import math
import random

class NeuralNet():
    def __init__(self, inputs, hidden, outputs, lr):
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

    # Calculate weighted sum
    def weighted_sum(self, inputs, weights, bias):
        return np.dot(inputs, weights.T) + bias

    # Step forward and back propagation processing
    def feed_forward(self, data):
        # Process first hidden layer
        h1 = self.weighted_sum(data, self.weights_ih, self.bias_h1)
        self.values_h1 = self.sigmoid(h1)

        # Process 2nd hidden layer
        h2 = self.weighted_sum(h1, self.weights_hh, self.bias_h2)
        self.values_h2 = self.sigmoid(h2)

        # Process output layer
        output = self.weighted_sum(h2, self.weights_ho, self.bias_o)
        self.values_o = self.sigmoid(output)
        print(self.values_o.mean())

    # Back propagation
    def back_prop(self, data, actual):
        # Get the difference between predicted and actual value
        loss = np.subtract(actual, self.values_o)

        o_delta = loss * self.sigmoid_der(self.values_o)

        o_error = np.dot(o_delta, self.weights_ho)
        h2_delta = o_error * self.sigmoid_der(self.values_h2)

        h2_error = np.dot(h2_delta, self.weights_hh.T)
        h1_delta = h2_error * self.sigmoid_der(self.values_h1)
        
        # Update the weights and biases between hidden layer 2 and output layer
        self.weights_ho += self.update_weights(self.values_h2, loss)
        self.bias_o += self.update_biases(loss)

        self.weights_hh += self.update_weights(self.values_h1, h2_delta)
        self.bias_h2 += self.update_biases(h2_delta)

        self.weights_ih += self.update_weights(data, h1_delta)
        self.bias_h1 += self.update_biases(h1_delta)

        # ----------------------------------------

        # h2_hh = self.sigmoid_der(self.weighted_sum(self.values_h2, self.weights_ho, self.bias_o)) 
        # h2_delta = np.dot(loss, self.weights_ho)
        # h2_gradient = np.dot(h2_hh.T, h2_delta)

        # # Update between two hidden nodes
        # self.weights_hh += self.update_weights(self.values_h1, h2_delta)
        # self.bias_h2 -= self.update_biases(h2_delta)

        # # Determine gradient of hidden layer 1
        # hidden1_error_der = self.sigmoid_der(self.values_h1)
        # hidden1_gradient = np.dot(h2_gradient, self.weights_hh.T) * hidden1_error_der

        # # Update weights and biases between inputs and first hidden layer
        # self.weights_ih += self.update_weights(data, hidden1_gradient)
        # self.bias_h1 += self.update_biases(hidden1_gradient)
        
        return loss

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

    def get_accuracy(self, error):
        # If the RMSE is more than .5 away, respond incorrectly classified
        acc = 0
        for i in error:
            if i < 0.5:
                acc += 1

        return acc
