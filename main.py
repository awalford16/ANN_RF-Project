from data_processing import Data
from plotting import Plot
from section3 import Models
from section4 import CrossVal
import pandas as pd
import pylab
import numpy as np

def main():
    print('---------- Section 1 ----------')
    # Create an instance of the data class which will store the csv data in a dataframe
    plant_data = Data('nuclear_plants.csv')
    
    # Normalise data to scale values
    # plant_data.norm_data()

    print(f"Mean: \n{plant_data.get_mean()}")
    print(f"Standard Deviation: \n{plant_data.get_stan_dev()}")
    print(f"Minimum: \n{plant_data.get_min()}")
    print(f"Maximum: \n{plant_data.get_max()}")
    print(f"Median: \n{plant_data.get_median()}")
    print(plant_data.get_missing_value_count())
    print(plant_data.get_feature_count())
    print(f"Variance: \n{plant_data.get_variance()}")

    plt = Plot()
    #print(plt.data_box_plot(plant_data.data, 'Status', 'Vibration_sensor_1'), pylab.show())
    #print(plt.data_density_plot(plant_data.data, 'Status', 'Vibration_sensor_2'), pylab.show())

    # Standardise the data
    plant_data.stand_data()
    #print(plant_data.data.head())

    # Convert status column to categorical data
    plant_data.cat_to_num('Status')

    print('---------- Section 3 ----------')
    # Split data into train and test
    train_x, train_y, test_x, test_y = plant_data.split_data(0.9, 'Status')

    models = Models(train_x, train_y, test_x, test_y)
    models.create_nn_model(500)
    models.train_nn()

    # Create Neural Network

    # Train Nerual Network with 500 nodes and 2 hidden layers

    # Apply test data to NN
    #models.test_nn()

    # Create Random forest with 1000 trees and 5 or 50 leaf nodes

    # Apply test data to random forest

    #plt.tree_count_plot(nodes, train_acc, test_acc)

    print('---------- Section 4 ----------')
    # Get the training segment of data including testing set
    # data = pd.concat([train_x, train_y], axis=1)
    # cv = CrossVal()
    # highest_acc = 0.5
    # best_model = ''

    # # Cross validate NN with 50, 500 and 1000 nodes
    # for i in [50, 500, 1000]:
    #     print(f'Cross validating NN with {i} nodes')
    #     accuracy = cv.cross_val('nn', 10, data, i)
    #     # Update the highest accuracy model to identify best value for nodes
    #     if accuracy > highest_acc:
    #         highest_acc = accuracy
    #         best_model = f'NN with {i} hidden nodes'

    # # Cross validate RF with 20, 500 and 10000 trees
    # for i in [20, 500, 10000]:
    #     print('Cross validating RF with {i} trees')
    #     accuracy = cv.cross_val('rf', 10, data, i)
    #     # Update the highest accuracy model to identify best value for trees
    #     if accuracy > highest_acc:
    #         highest_acc = accuracy
    #         best_model = f'RF with {i} trees'

    # print(f'Best Model: {best_model}')

if __name__ == '__main__':
    main()