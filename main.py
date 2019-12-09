from data_processing import Data
from plotting import Plot
from section3 import Models
import pylab
import numpy as np

def main():
    print('---------- Section 1 -----------')
    # Create an instance of the data class which will store the csv data in a dataframe
    plant_data = Data('nuclear_plants.csv')
    
    # Normalise data to scale values
    # plant_data.norm_data()

    # print(f"Mean: \n{plant_data.get_mean()}")
    # print(f"Standard Deviation: \n{plant_data.get_stan_dev()}")
    # print(f"Minimum: \n{plant_data.get_min()}")
    # print(f"Maximum: \n{plant_data.get_max()}")
    # print(f"Median: \n{plant_data.get_median()}")
    # print(plant_data.get_missing_value_count())
    # print(plant_data.get_feature_count())
    # print(f"Variance: \n{plant_data.get_variance()}")

    plt = Plot()
    #print(bx_plt.data_box_plot(plant_data.data, 'Status', 'Vibration_sensor_1'), pylab.show())
    #print(bx_plt.data_density_plot(plant_data.data, 'Status', 'Vibration_sensor_2'), pylab.show())

    # Standardise the data
    plant_data.stand_data()
    #print(plant_data.data.head())

    # Convert status column to categorical data
    plant_data.cat_to_num('Status')

    print('---------- Section 3 ----------')
    # Split data into train and test
    train_x, train_y, test_x, test_y = plant_data.split_data(0.9, 'Status')

    models = Models(train_x, train_y, test_x, test_y)

    num_trees = [10, 50, 100, 1000, 5000, 10000]
    train_acc = np.zeros(len(num_trees))
    test_acc = np.zeros(len(num_trees))

    for i, tree in enumerate(num_trees):
        # Create the models
        models.create_forest_model(tree)
    #models.train_nn()
    #models.test_nn()
    
        models.train_forest()
        train, test = models.test_forest()

        # Add accuracies to array for plotting
        train_acc[i] = train
        test_acc[i] = test
        print(f'Training Error with {tree} Trees: {train}')
        print(f'Testing Error with {tree} Trees: {test}')

    plt.tree_count_plot(num_trees, train_acc, test_acc)

    print('---------- Section 4 ----------')

if __name__ == '__main__':
    main()