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
        # Get number of missing values in each column
        return self.data.isnull().sum()

    # Get the number of features
    def get_feature_count(self):
        # Get the number of columns in each observation (Use first observation for column count)
        return self.data.count(axis=1)[0]

    def get_variance(self):
        # Pandas functionality to get the variation of each column
        return self.data.var()

    def norm_data(self):
        # Ignore data that is not numerical
        data = self.data.select_dtypes(include=[np.number])
        self.data = (data-data.min())/(data.max()-data.min())

    # Standardise the data
    def stand_data(self):
        data = self.data.select_dtypes(include=[np.number])
        # Externally store status data
        status_data = self.data['Status']
        self.data = ((data-data.mean())/(data.std(axis=0)))
        # Append status data after standardisation
        self.data['Status'] = status_data

    # Split the data by a requested %
    def split_data(self, train_split):
        # Shuffle the data
        data = self.data.sample(frac=1)

        # Get percentages of data and assign as training and test sets
        percent = int((data.shape[0]) * train_split)
        train = data[:percent]
        test = data[percent:]

        # Return training and test splits, separating the target variable
        return train.loc[:, train.columns != 'Status'], train['Status'], test.loc[:, test.columns != 'Status'], test['Status']
