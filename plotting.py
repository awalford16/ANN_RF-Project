import pandas as pd
import matplotlib.pyplot as plt
import os

class Plot:
    def data_box_plot(self, data, col1, col2):
        # Group the data based on the first column (status)
        status_data = data.groupby(col1)
        return status_data.boxplot(column=col2)

    def data_density_plot(self, data, col1, col2):
        # Group the data and use the data from col2 of each group
        data.groupby(col1)[col2].plot(kind='density', legend=True)

    def nn_error_plot(self, error):
        plt.figure()
        plt.plot(error)
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join('images', 'epoch_error.png'))