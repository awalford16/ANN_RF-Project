from section3 import Models
import numpy as np
import pandas as pd

class CrossVal():
    def __init__(self, folds):
        self.folds = folds

    # Loop through x folds and train model
    def cross_val(self, model_choice, data, count):
        # Create array to store accuracy score for each fold
        accuracies = np.zeros(self.folds)

        # Loop through folds of data
        for x in range(self.folds):
            # Obtain fold data
            train, test = self.get_fold(x, data)
            model = Models(train.loc[:, train.columns != 'Status'], train['Status'], test.loc[:, test.columns != 'Status'], test['Status'])
        
            # Select which model to train
            if model_choice == 'rf':
                # Store accuracy for fold
                accuracies[x] = self.cross_val_rf(model, train, test, count)
            elif model_choice == 'nn':
                accuracies[x] = self.cross_val_nn(model, train, test, count)
            else:
                print(f'{model_choice} not recognised as a model.')

        print(f'Mean Accuracy of {model_choice.upper()}: {accuracies.mean():.2f}')

        # Return the average mean accuracy
        return accuracies.mean()

    # Train and test random forest with segmented data
    def cross_val_rf(self, model, train, test, tree_count):
        # Create a random forest with x trees
        model.create_forest_model(tree_count)

        # Train model with segment of data and get accuracy with fold
        model.train_forest()
        _, test = model.test_forest()
        return test

    def cross_val_nn(self, model, train, test, node_count):
        # Initialise NN with x nodes
        model.create_nn_model(node_count, 0.0001)

        # Train the neural network with training set
        model.train_nn(False)
        return model.test_nn()

    def get_fold(self, fold, data):
        # Get approximate size of each fold
        size = int(len(data.values) / self.folds)
        start = size * fold

        # Get test portion of data
        test = data[start:(start+size)]

        # Get training portion of data
        train = pd.concat([data[:start], data[(start+size):]], axis=0)

        return train, test

