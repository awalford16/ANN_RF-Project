from data_processing import Data
from plotting import Plot
from neural_net import NeuralNet
import pylab
import numpy as np

def main():
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

    #bx_plt = Plot()
    #print(bx_plt.data_box_plot(plant_data.data, 'Status', 'Vibration_sensor_1'), pylab.show())
    #print(bx_plt.data_density_plot(plant_data.data, 'Status', 'Vibration_sensor_2'), pylab.show())

    # Standardise the data
    plant_data.stand_data()
    #print(plant_data.data.head())

    # Convert status column to categorical data
    plant_data.cat_to_num('Status')

    # Split data into train and test
    train_x, train_y, test_x, test_y = plant_data.split_data(0.9, 'Status')

    # Create a neural net with 2 hidden layers using standardised nuclear plant data
    net = NeuralNet(train_x.shape[1], 250, 2, 1, 0.2)

    # Create empty array to store results
    results = np.zeros((1, 513))

    for m in range(5):
        # for i in range(len(train_x.values)):
        net.process(train_x.values[-1])
        error = net.process_error(train_y.values[-1])
        print((net.values[:12], net.values[-1], train_y.values[-1], error))

if __name__ == '__main__':
    main()