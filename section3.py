from neural_net import NeuralNet
from random_forest import RandomForest
import numpy as np
from plotting import Plot

class Models:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y

        # Create a neural net with 2 hidden layers using standardised nuclear plant data
        self.nn = NeuralNet(self.train_x.shape[1], 50, 2, 1, 0.7)

        # Initialise random forest model
        self.rf = RandomForest(1000, 5)

    # Use the NeuralNetwork class to train a NN
    def train_nn(self):
        EPOCHS = 10
        total_error = np.zeros(EPOCHS)

        for m in range(EPOCHS):
            for i in range(len(self.train_x.values)):
                print(f'{i + 1} OUT OF {len(self.train_x.values)}')
                self.nn.feed_forward(self.train_x.values[i])
                error = self.nn.back_propagation(self.train_y.values[i])
                
                total_error[m] += error

                output = (self.nn.values[:12], self.nn.values[-1], self.train_y.values[-1], error)
                print (output)

        plt = Plot()
        plt.nn_error_plot(total_error)
        # Get the weights adjusted by training
        return self.nn.weights

    # Train a random forest model
    def train_forest(self):
        self.rf.train_forest(self.train_x, self.train_y)


    