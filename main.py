from data_processing import Data
from plotting import Plot
import pylab

def main():
    dp = Data('nuclear_plants.csv')
    
    # Normalise data to scale values
    # dp.norm_data()

    # print(f"Mean: \n{dp.get_mean()}")
    # print(f"Standard Deviation: \n{dp.get_stan_dev()}")
    # print(f"Minimum: \n{dp.get_min()}")
    # print(f"Maximum: \n{dp.get_max()}")
    # print(f"Median: \n{dp.get_median()}")
    # print(dp.get_missing_value_count())
    # print(dp.get_feature_count())
    # print(f"Variance: \n{dp.get_variance()}")

    bx_plt = Plot()
    print(bx_plt.data_box_plot(dp.data, 'Status', 'Vibration_sensor_1'), pylab.show())
    print(bx_plt.data_density_plot(dp.data, 'Status', 'Vibration_sensor_2'), pylab.show())

if __name__ == '__main__':
    main()