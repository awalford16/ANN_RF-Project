from neural_net_2 import NeuralNet
from random_forest import RandomForest
import numpy as np
from plotting import Plot
import math

class Models:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x.values
        self.train_y = train_y.values
        self.test_x = test_x.values
        self.test_y = test_y.values

    def create_forest_model(self, trees):
        # Initialise random forest model
        self.rf = RandomForest(trees, 50)

    def create_nn_model(self, hidden_nodes, lr):
        # Create a neural net with 2 hidden layers using standardised nuclear plant data
        self.nn = NeuralNet(self.train_x.shape[1], hidden_nodes, 1, lr)

    # divide dataset into mini batches when one epoch is too large in one go
    def get_mini_batches(self, train, test, batch_size):
        batch_count = math.ceil(len(train) / batch_size)
        x_batches = np.zeros((batch_count, batch_size, 12))
        y_batches = np.zeros((batch_count, batch_size, 1))

        for batch in range(batch_count):
            if (batch * batch_size) + batch_size > len(train):
                batch_size = len(train) - (batch * batch_size)

            # Initialise start and finish index for batch
            start = batch * batch_size
            finish = start + batch_size

            for i in range(batch_size):
                x_batches[batch][i] = train[start:finish][i]
                y_batches[batch][i] = test[start:finish][i]

        return x_batches, y_batches

    # Use the NeuralNetwork class to train a NN
    def train_nn(self, plot):
        EPOCHS = 10
        total_acc = np.zeros(EPOCHS)

        # Run for multiple epochs to improve accuracy
        for m in range(EPOCHS):
            self.nn.feed_forward(self.train_x)

            # Update the weights based on the data error
            self.nn.back_prop(self.train_x, self.train_y.reshape(self.train_y.shape[0],1))

            acc = self.nn.get_accuracy(self.train_y)
   
            total_acc[m] = acc
            print(f'Accuracy: {total_acc[m]}')

        if plot:
            plt = Plot()
            plt.nn_acc_plot(total_acc)

    # Test the NN
    def test_nn(self):
        #total = 0
        self.nn.feed_forward(self.test_x)
        total = self.nn.get_accuracy(self.test_y)

        return total

    # Train a random forest model
    def train_forest(self):
        self.rf.train_forest(self.train_x, self.train_y)

    def test_forest(self):
        train, test = self.rf.get_forest_accuracy(self.train_x, self.train_y, self.test_x, self.test_y)
        return train, test
    