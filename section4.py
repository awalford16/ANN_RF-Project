from section3 import Models
import numpy as np
import pandas as pd

class CrossVal():
    # Loop through x folds and train model
    def cross_val(self, model_choice, folds, data, count):
        accuracies = np.zeros(folds)
        for x in range(folds):
            # Segment data into x folds
            train, test = self.get_fold(x, folds, data)
            model = Models(train.loc[:, train.columns != 'Status'], train['Status'], test.loc[:, test.columns != 'Status'], test['Status'])
        
            # Train model with each fold
            if model_choice == 'rf':
                # Pass in train and test segments
                accuracies[x] = self.cross_val_rf(model, train, test, count)
            elif model_choice == 'nn':
                accuracies[x] = self.cross_val_nn(model, train, test, count)
            else:
                print(f'{model_choice} not recognised as a model.')
            
            print(f'Fold {x + 1} accuracy: {accuracies[x]}')

        print(f'Mean Accuracy of {model_choice}: {accuracies.mean()}')

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
        model.create_nn_model(node_count)

        # Train the neural network with training set
        model.train_nn()
        return model.test_nn()

    def get_fold(self, fold, k, data):
        # Get approximate size of each fold
        size = int(len(data.values) / k)
        start = size * fold

        # Get test portion of data
        test = data[start:(start+size)]

        # Get training portion of data
        train = pd.concat([data[:start], data[(start+size):]], axis=0)

        return train, test

