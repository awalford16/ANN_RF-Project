from sklearn import neural_network

class NeuralNet:
    def __init__(self, hidden, lr, epochs):
        # Initialise a network with 2 hidden layers
        self.nn = neural_network.MLPClassifier(hidden_layer_sizes=(hidden, hidden), activation='logistic', learning_rate_init=lr, n_iter_no_change=epochs)

    def train_nn(self, x, y):
        self.nn.fit(x, y)

    def test_nn(self, test_x, test_y):
        return self.nn.score(test_x, test_y)
