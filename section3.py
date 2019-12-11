from neural_net_2 import NeuralNet
from random_forest import RandomForest
import numpy as np
from plotting import Plot

class Models:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x.values
        self.train_y = train_y.values
        self.test_x = test_x.values
        self.test_y = test_y.values

    def create_forest_model(self, trees):
        # Initialise random forest model
        self.rf = RandomForest(trees, 50)

    def create_nn_model(self, hidden_nodes):
        # Create a neural net with 2 hidden layers using standardised nuclear plant data
        self.nn = NeuralNet(self.train_x, self.train_x.shape[1], hidden_nodes, 1, 0.3)

    # Use the NeuralNetwork class to train a NN
    def train_nn(self):
        EPOCHS = 10
        total_error = np.zeros(EPOCHS)
        # predictions = np.zeros(len(self.train_x))

        for m in range(EPOCHS):
            total_acc = 0
            print(f'Epoch: {m + 1}')
            self.nn.feed_forward(self.train_x)

            # Update the weights based on the data error
            total_error[m] = self.nn.back_prop(self.train_x, self.train_y)

            total_acc += self.nn.get_accuracy(total_error[m], self.train_y)

            print(f'Accuracy: {total_acc}')

        plt = Plot()
        plt.nn_error_plot(total_error)

    # Test the NN
    def test_nn(self):
        #total = 0
        self.nn.feed_forward(self.test_x)
        #total += self.nn.get_accuracy(self.nn.values[-1], self.test_y[i])

        # acc = (total / len(self.test_x))
        #return acc

    # Train a random forest model
    def train_forest(self):
        self.rf.train_forest(self.train_x, self.train_y)

    def test_forest(self):
        train, test = self.rf.get_forest_error(self.train_x, self.train_y, self.test_x, self.test_y)
        return train, test
    