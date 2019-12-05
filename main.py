from data_processing import Data
import os

def main():
    dp = Data('nuclear_plants.csv')
    
    # Normalise data to scale values
    # dp.norm_data()

    print(dp.data.head())
    print(dp.get_mean())
    print(dp.get_stan_dev())
    print(dp.get_min())
    print(dp.get_max())
    print(dp.get_median())
    print(dp.get_missing_value_count())
    print(dp.get_feature_count())

if __name__ == '__main__':
    main()