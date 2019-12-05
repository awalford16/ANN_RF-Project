import pandas as pd

class Plot:
    def data_box_plot(self, data, col1, col2):
        status_data = data.groupby(col1)
        return status_data.boxplot(column=col2)

    def data_density_plot(self, data, col1, col2):
        # Group by status and plot based on column2
        data.groupby(col1)[col2].plot(kind='density', legend=True)