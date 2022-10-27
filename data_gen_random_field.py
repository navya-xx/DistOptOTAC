"""Random Field Estimation problem.

Data sequence generator class.
- Iteratively generate: Features, target-values, network graph
"""

import numpy as np
import os
import csv


class Oceanograph_realdata_winter:
    """Parse data from oceanography dataset.

    Only data for winter season is used.

    Data obtained from : https://www.ncei.noaa.gov/access/gulf-of-mexico-climate/gulf-data.html
    """

    def __init__(self, num_agents, field_size, noise_scale, depth_limit=100, seed=None, partition_data=True) -> None:
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.N = num_agents
        self.field_size = field_size
        self.rng = np.random.default_rng(seed)
        self.depth_limit = depth_limit
        self.temp_data_winter()
        if partition_data:
            self.train_data, self.test_data = self.data_partition()
        else:
            self.train_data = self.data
            self.test_data = self.data
        self.data_len = self.data.shape[0]
        self.noise_scale = noise_scale
        
    def temp_data_winter(self):
        """Read Oceanography temperature data - winter statistical mean.
        """

        filename = self.curr_dir + "/OceanographyData/winter_data_dlim%d.npz" % (self.depth_limit)
        if os.path.isfile(filename):
            saved_data = np.load(filename)
            self.data = saved_data['data']
            self.depths = saved_data['depths']
        else:
            csv_file = self.curr_dir + "/OceanographyData/gom_decav_t13mn10.csv"
            entire_data = []

            with open(csv_file, 'r') as f:
                fdata = csv.reader(f, delimiter=',')
                i = 0
                for frow in fdata:
                    if i == 1:
                        depths = np.array([0] + frow[3:104], dtype=int)
                        depths = np.delete(depths, np.argwhere(depths > self.depth_limit).flatten())
                        self.depths = depths
                    elif i > 1:
                        tmp = [float(frow[0]), float(frow[1])]
                        for j in range(depths.size):
                            if j + 2 > len(frow) - 1: # no data in this row
                                continue
                            if frow[j+2] != "":  # data point empty
                                entire_data.append(tmp + [int(depths[j])] + [float(frow[j+2])])

                    i = i + 1

            self.data = self.normalize_to_field(np.array(entire_data, dtype=float))
            np.savez(filename, data=self.data, depths=self.depths, allow_pickle=False)


    def normalize_to_field(self, data):
        mins, maxs = np.min(data[:, :-1], axis=0, keepdims=True), np.max(data[:, :-1], axis=0, keepdims=True)
        # field is a cube
        data[:, :-1] = (data[:, :-1] - mins)/(maxs - mins) * self.field_size
        # temp between [0, 1]
        data[:, -1] = (data[:, -1] - np.min(data[:, -1]))/(np.max(data[:, -1]) - np.min(data[:, -1]))
        return data

    def data_partition(self, training_part=0.7):
        # divide data into training and testing
        # enable same partition of data
        data = self.data
        data_len = data.shape[0]
        train_len = int(data_len * training_part)
        train_ids = self.rng.choice(np.arange(data_len), train_len, replace=False)
        test_ids = np.delete(np.arange(data_len), train_ids)
        train_data = data[train_ids, :]
        test_data = data[test_ids, :]
        return train_data, test_data

    def next_measurements(self):
        data = self.train_data
        data_len = data.shape[0]
        next_ids = self.rng.choice(np.arange(data_len), self.N, replace=False)
        noise_vals = self.rng.standard_normal(data[next_ids, -1].shape) * self.noise_scale
        return data[next_ids, :-1], data[next_ids, -1] + noise_vals


if __name__ == "__main__":

    N = 100
    field_size = 1000
    data_gen = Oceanograph_realdata_winter(N, field_size, seed=123)
    print(data_gen.depths)
    print(np.unique(data_gen.data[:, 2]))