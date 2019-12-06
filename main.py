from data_processing import Data
from plotting import Plot
import pylab

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

    bx_plt = Plot()
    print(bx_plt.data_box_plot(plant_data.data, 'Status', 'Vibration_sensor_1'), pylab.show())
    print(bx_plt.data_density_plot(plant_data.data, 'Status', 'Vibration_sensor_2'), pylab.show())

if __name__ == '__main__':
    main()