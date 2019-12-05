from data_processing import DataProcessing
import os

def main():
    dp = DataProcessing(os.path.join('data', 'nuclear_plants.csv'))
    
    # Normalise data to scale values
    # dp.norm_data()

    print(dp.data.head())
    print(dp.get_mean())
    print(dp.get_stan_dev())
    print(dp.data.min())
    print(dp.data.max())
    print(dp.data.median())

if __name__ == '__main__':
    main()