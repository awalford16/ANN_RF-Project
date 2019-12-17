from section3 import Models
import numpy as np
import pandas as pd

class CrossVal():
    def __init__(self, folds):
        self.folds = folds

    # Loop through x folds and train model
    def cross_val(self, model_choice, data, count):
        # Create array to store accuracy score for each fold
        train_accuracies = np.zeros(self.folds)
        test_accuracies = np.zeros(self.folds)

        # Loop through folds of data
        for x in range(self.folds):
            # Obtain fold data
            train, val = self.get_fold(x, data)
            train_x = train.loc[:, train.columns != 'Status']
            train_y = train['Status']
            val_x = val.loc[:, val.columns != 'Status']
            val_y = val['Status']
            model = Models(train_x, train_y, val_x, val_y)
        
            # Select which model to train
            if model_choice == 'rf':
                # Store accuracy for fold
                train_accuracies[x], test_accuracies[x] = self.cross_val_rf(model, count)
            elif model_choice == 'nn':
                train_accuracies[x], test_accuracies[x]  = self.cross_val_nn(model, count)
            else:
                print(f'{model_choice} not recognised as a model.')

        print(f'Mean Training Accuracy of {model_choice.upper()}: {train_accuracies.mean():.2f}')
        print(f'Mean Testing Accuracy of {model_choice.upper()}: {test_accuracies.mean():.2f}')

        # Return the average mean accuracy
        return test_accuracies.mean()

    # Train and test random forest with segmented data
    def cross_val_rf(self, model, tree_count):
        # Create a random forest with x trees
        model.create_forest_model(tree_count)

        # Train model with segment of data and get accuracy with fold
        model.train_forest()
        
        return model.test_forest()

    def cross_val_nn(self, model, node_count):
        # Initialise NN with x nodes and a learning rate of 0.0001
        model.create_nn_model(node_count, 0.0001)

        # Train the neural network with 100 epochs
        train = model.train_nn(False, 150)

        # Return the train and testing accuracies
        return train, model.test_nn()

    def get_fold(self, fold, data):
        # Get approximate size of each fold
        size = int(len(data.values) / self.folds)
        start = size * fold

        # Get validation portion of data
        val = data[start:(start+size)]

        # Get training portion of data
        train = pd.concat([data[:start], data[(start+size):]], axis=0)

        return train, val

