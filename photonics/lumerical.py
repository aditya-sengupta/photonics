import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

def row_to_list(l, t):
    l = l.split(",")
    l[-1] = l[-1][:-1]
    return list(map(t, l))

def list_to_polars(l):
    d = {}
    for terms in zip(*l):
        d[terms[0]] = terms[1:]

    return pl.DataFrame(d)

# if speed ever becomes a bottleneck there's this: 
# https://stackoverflow.com/questions/77285192/reading-multiple-polars-dataframes-from-a-single-csv-file/77288818?noredirect=1#comment136256836_77288818
def read_csv_and_split_tables(file_path):
    with open(file_path, 'r') as file:
        table_data = []
        current_table = []
        for line in file:
            if not current_table:
                current_table.append(row_to_list(line, str))
            elif line.strip():
                current_table.append(row_to_list(line, float))
            else:
                if current_table:
                    table_data.append(current_table)
                current_table = []

        if current_table:
            table_data.append(current_table)
    
    return list(map(list_to_polars, table_data))

class LumericalData:
    def __init__(self, xy, interp_step=5e-8):
        self.xr = xy['x']
        self.yr = xy['y']
        self.compute_grid(interp_step)
    
    def compute_grid(self, interp_step):
        r = np.arange(self.xr.min(), self.xr.max(), interp_step)
        self.x, self.y = np.meshgrid(r, r, indexing='ij')
        self.r2 = self.x ** 2 + self.y ** 2
        self.r = np.sqrt(self.r2)

    def regrid(self, data):
        interp_data = griddata((self.xr, self.yr), data, (self.x, self.y), fill_value=data[0])
        # fill_value of data[0] isn't perfect, but should do fine for Lumerical data
        # where this'll be a corner, where we should expect no physics
        return (self.x, self.y, interp_data)

    def plot(self, data, **kwargs):
        """
        Takes in 1D data defined over (self.xr, self.yr), regrids it, and plots it as a colormesh over (self.x, self.y).
        """
        plt.pcolormesh(*self.regrid(data), **kwargs)
        plt.colorbar()