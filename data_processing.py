import pandas as pd
import numpy as np
import os

class Data:
    def __init__(self, csv_file):
        # Create data property to store CSV data within a dataframe
        self.data = self.read_data(csv_file)

    def read_data(self, file_name):
        df = pd.read_csv(os.path.join('data', file_name))
        return df

    def get_mean(self):
        # Get the mean values for each column (axis = 0)
        return self.data.mean(axis=0)

    def get_stan_dev(self):
        # Get the standard deviation of each column within dataset
        return self.data.std(axis=0)

    # Use pandas functionality to get min and max values
    def get_min(self):
        return self.data.min()

    def get_max(self):
        return self.data.max()

    # Get median using pandas functionality
    def get_median(self):
        return self.data.median()

    # Identify missing values within dataset
    def get_missing_value_count(self):
        # Get number of missing values in each column (axis = 0)
        return self.data.isnull().sum()

    def norm_data(self):
        data = self.data.select_dtypes(include=[np.number])
        self.data = (data-data.min())/(data.max()-data.min())