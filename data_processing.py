import pandas as pd
import numpy as np

class DataProcessing:
    def __init__(self, csv_file):
        self.data = self.read_data(csv_file)

    def read_data(self, csv_file):
        df = pd.read_csv(csv_file)
        return df

    def get_mean(self):
        # Get the mean values for each column (axis = 0)
        return self.data.mean(axis=0)

    def get_stan_dev(self):
        # Get the standard deviation of each column within dataset
        return self.data.std(axis=0)

    def norm_data(self):
        data = self.data.select_dtypes(include=[np.number])
        self.data = (data-data.min())/(data.max()-data.min())