from data_processing import Data
from plotting import Plot
from section3 import Models
from section4 import CrossVal
from neural_net_scikit import NeuralNet
import pandas as pd
import pylab
import numpy as np

def main():
    print('---------- Section 1 ----------')
    # Create an instance of the data class which will store the csv data in a dataframe
    plant_data = Data('nuclear_plants.csv')
    
    # Normalise data to scale values
    # plant_data.norm_data()
    print(f'Size of data: {plant_data.get_size()}')
    print(f'Data Types: \n{plant_data.get_data_type()}')
    print(f'Number of Samples for each Category: {plant_data.get_cat_count("Status")}')

    print(f"Mean: \n{plant_data.get_mean()}")
    print(f"Standard Deviation: \n{plant_data.get_stan_dev()}")
    print(f"Minimum: \n{plant_data.get_min()}")
    print(f"Maximum: \n{plant_data.get_max()}")
    print(f"Median: \n{plant_data.get_median()}")
    print(plant_data.get_missing_value_count())
    print(plant_data.get_feature_count())
    print(f"Variance: \n{plant_data.get_variance()}")

    #plt = Plot()
    #print(plt.data_box_plot(plant_data.data, 'Status', 'Vibration_sensor_1'), pylab.show())
    #print(plt.data_density_plot(plant_data.data, 'Status', 'Vibration_sensor_2'), pylab.show())

    # Standardise the data
    plant_data.stand_data()
    #print(plant_data.data.head())

    # Convert status column to categorical data
    plant_data.cat_to_num('Status')

    print('---------- Section 3 ----------')
    # Split data into train and test based on target variable Status
    train_x, train_y, test_x, test_y = plant_data.split_data(0.9, 'Status')

    # # for e in epochs:
    # models = Models(train_x, train_y, test_x, test_y)
    
    # # Create Neural Network
    # models.create_nn_model(500, 0.0001)

    # # Train Nerual Network with 500 nodes and 2 hidden layers
    # models.train_nn(True, 150)

    # # Apply test data to NN
    # acc = models.test_nn()
    # print(f'NN Testing Accuracy: {acc}')

    # Create Random forest with 1000 trees and 5 or 50 leaf nodes
    # models.create_rf_model(5)

    # Apply test data to random forest

    #plt.tree_count_plot(nodes, train_acc, test_acc)

    print('---------- Section 4 ----------')
    # Get the training segment of data including testing set
    # data = pd.concat([train_x, train_y], axis=1)
    # cv = CrossVal(10)
    # highest_acc = 0.5
    # best_model = ''

    # # Cross validate NN with 50, 500 and 1000 nodes
    # for i in [50, 500, 1000]:
    #     print(f'Cross validating NN with {i} nodes')
    #     test_acc = cv.cross_val('nn', data, i)
    #     # Update the highest accuracy model to identify best value for nodes
    #     if test_acc > highest_acc:
    #         highest_acc = test_acc
    #         best_model = f'NN with {i} hidden nodes'

    # # Cross validate RF with 20, 500 and 10000 trees
    # for i in [20, 500, 10000]:
    #     print(f'Cross validating RF with {i} trees')
    #     test_acc = cv.cross_val('rf', data, i)
    #     # Update the highest accuracy model to identify best value for trees
    #     if test_acc > highest_acc:
    #         highest_acc = test_acc
    #         best_model = f'RF with {i} trees'

    # print(f'Best Model: {best_model}')

if __name__ == '__main__':
    main()