import random
import numpy as np
import matplotlib.pyplot as plt


class TestDataSet:

    def __init__(self, k=3, points=100, dim_x=1000, dim_y=1000, deviation=25):
        self.k = k
        self.number_of_points = points
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.deviation = deviation
        self.centers = np.zeros((self.k, 2))
        self.all_points = 0

        # Generate center random points
        # Center points are also in all_points now
        self.generate_random_center_points()
        self.generate_cluster_points()

    def __repr__(self):
        return str(self.all_points)

    def generate_random_center_points(self):
        for i in range(self.k):
            x = random.randint(0, self.dim_x)
            y = random.randint(0, self.dim_y)
            self.centers[i] = np.array([x, y])
            self.all_points = self.centers

    def generate_cluster_points(self):
        points_per_cluster = int(self.number_of_points / self.k) - 1
        # if result is not even
        # remainding_nmbr_of_points = self.number_of_points - (self.k * points_per_cluster)

        for i in range(self.k):
            for j in range(points_per_cluster):
                x_new = int(np.random.normal(self.centers[i][0], self.deviation))
                x_new = self.dim_x if x_new > self.dim_x else x_new
                x_new = 0 if x_new < 0 else x_new

                y_new = int(np.random.normal(self.centers[i][1], self.deviation))
                y_new = self.dim_y if y_new > self.dim_y else y_new
                y_new = 0 if y_new < 0 else y_new
                self.all_points = np.append(self.all_points, np.array([[x_new, y_new]]), axis=0)

    def plot(self):
        plt.plot(data_set.all_points[:, 0], data_set.all_points[:, 1], 'bo')
        plt.show()
