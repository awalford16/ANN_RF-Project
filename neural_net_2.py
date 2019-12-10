from torch import nn

# An improved approach to the original neural network
class NeuralNet(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.input_nodes = nn.Linear(inputs, hidden)
        self.hidden_nodes = nn.Linear(hidden, hidden)
        self.output_nodes = nn.Linear(hidden, outputs)

        # Define activation function
        self.sigmoid = nn.Sigmoid()


    def step_forward(self, x):
        # Compute nodes from input to first hidden layer
        x = self.sigmoid(self.input_nodes(x))

        # Compute nodes between hidden layers
        x = self.sigmoid(self.hidden_nodes(x))

        # Compute nodes between second hidden layer and output
        x = self.sigmoid(self.output_nodes(x))
