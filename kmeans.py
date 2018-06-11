from pip._vendor import colorama

from kmeans_test_data import *
import math
import time

data = TestDataSet()


class KMeans:
    def __init__(self, k=50, points=10000, dim_x=10000, dim_y=10000, deviation=200, oversampling_factor=25):
        self.k = k
        self.nmbr_of_points = points
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.deviation = deviation
        self.oversampling_factor = oversampling_factor
        self.center_points = []
        self.d_i = []
        self.pdf = []
        self.clusters = []
        self.plot_colors = ['ro', 'co', 'mo', 'yo', 'ko', 'wo', 'bo', 'go']
        # Generate test data
        self.data = TestDataSet(k=self.k, points=self.nmbr_of_points, dim_x=self.dim_x, dim_y=dim_y,
                                deviation=self.deviation)
        # update number of points
        self.nmbr_of_points = self.data.all_points.shape[0]
        # np.seterr(divide='ignore', invalid='ignore')

    def initialize_centroids_random(self):
        # Generate k random centroids
        self.center_points = []
        self.center_points = [self.data.all_points[int(random.randint(0, self.nmbr_of_points))] for k in range(self.k)]

    def initialize_centroids_plus_plus(self):
        self.center_points = []
        # Choose random center point (step 1)
        c_new = self.data.all_points[int(random.randint(0, self.nmbr_of_points))]
        self.center_points.append(c_new)

        # Calculate more points
        for j in range(self.k - 1):
            # Calculate squared eucl. distance to center point for every point (step 2)
            for i in range(self.data.all_points.shape[0]):
                # No distances calculated until now
                if len(self.d_i) < self.data.all_points.shape[0]:
                    self.d_i.append(np.linalg.norm(c_new - self.data.all_points[i]) ** 2)
                # Checking for the min. distance
                else:
                    for centroid in self.center_points:
                        # Comparing old distance with new distance
                        if np.linalg.norm(centroid - self.data.all_points[i]) ** 2 < self.d_i[i]:
                            self.d_i[i] = np.linalg.norm(centroid - self.data.all_points[i]) ** 2

            # Summing up all distances
            d_x_sum = sum(self.d_i)

            # Calculate PDF
            self.pdf = [(d / d_x_sum) for d in self.d_i]

            # Choose value with respect to the probabilities
            c_new = self.data.all_points[np.random.choice(np.arange(len(self.pdf)), p=self.pdf, size=1)][0]
            self.center_points.append(c_new)

    def initialize_centroids_parallel(self, psi=0):
        self.center_points = []
        # Sample a point uniformly at random from X
        self.center_points = [self.data.all_points[int(random.randint(0, self.nmbr_of_points))]]
        # Calculate the cost psi
        psi = [psi + np.linalg.norm(self.center_points[0] - self.data.all_points[i]) ** 2 for i in
               range(self.data.all_points.shape[0])]
        psi = sum(psi)
        number_of_rounds = round(math.log(psi))
        for r in range(5):
            # Calculate distance to closest center for each point, and sum up
            self._calculate_distance_to_closest_centroid()
            new_costs = sum(self.d_i)
            for i in range(self.data.all_points.shape[0]):
                probability = (self.oversampling_factor * self.d_i[i]) / new_costs
                # sample point
                if random.uniform(0, 1) <= probability:
                    self.center_points.append(self.data.all_points[i])
        # Choose from many centroids
        self._classify_points()
        occurrences = {str(i): self.clusters.count(i) for i in range(len(self.center_points))}
        new_centroids = []
        # Take centroids with highest occurrence
        for i in range(self.k):
            # Delete item and save it into center points:
            # del self.center_points[index]
            maximum_occurrences = max(occurrences, key=occurrences.get)
            del occurrences[maximum_occurrences]
            new_centroids.append(self.center_points[int(maximum_occurrences)])
        self.center_points = new_centroids

    def start_clustering(self):
        number_of_iterations = 1

        while True:
            self._classify_points()
            # self.plot()
            new_centroids = self._recalculate_centroids()
            chk = True
            for i in range(len(new_centroids)):
                if (new_centroids[i][0] != self.center_points[i][0]) and (
                        new_centroids[i][1] != self.center_points[i][1]):
                    chk = False
            if chk is True:
                print("Number of iterations needed: ".format(number_of_iterations))
                print(number_of_iterations)
                break
            self.center_points = new_centroids
            number_of_iterations += 1

    def _classify_points(self):
        self.clusters = []
        for i in range(self.data.all_points.shape[0]):
            # Calculate the distance from each point to the centroids
            smallest_distance = np.linalg.norm(self.center_points[0] - self.data.all_points[i])
            k_estimated = 0
            for idx, center in enumerate(self.center_points):
                distance = np.linalg.norm(center - self.data.all_points[i])
                if distance < smallest_distance:
                    smallest_distance = distance
                    k_estimated = idx
            self.clusters.append(k_estimated)

    def _recalculate_centroids(self):
        centroids = []
        for i in range(self.k):
            centroid = np.zeros((1, 2))
            positions = [j for j, s in enumerate(self.clusters) if i is s]
            for p in positions:
                centroid = centroid + self.data.all_points[p]
            if len(positions) != 0:
                centroids.append(np.divide(centroid, len(positions))[0])
        return centroids

    def _calculate_distance_to_closest_centroid(self):
        for i in range(self.data.all_points.shape[0]):
            # No distances calculated until now
            if len(self.d_i) < self.data.all_points.shape[0]:
                self.d_i.append(np.linalg.norm(self.center_points[0] - self.data.all_points[i]) ** 2)
            # Checking for the min. distance
            else:
                for centroid in self.center_points:
                    # Comparing old distance with new distance
                    if np.linalg.norm(centroid - self.data.all_points[i]) ** 2 < self.d_i[i]:
                        self.d_i[i] = np.linalg.norm(centroid - self.data.all_points[i]) ** 2

    def plot(self):
        # Plot all points
        plt.plot(self.data.all_points[:, 0], self.data.all_points[:, 1], 'bo')

        # Plot clusters with different colors
        for i in range(self.k):
            rgb = self.plot_colors[int(random.uniform(0, 6))]
            positions = [j for j, s in enumerate(self.clusters) if i is s]
            for position in positions:
                plt.plot(self.data.all_points[position, 0], self.data.all_points[position, 1], rgb)

        # Plot centers
        for p in self.center_points:
            plt.plot(p[0], p[1], 'g^')

        plt.show()


if __name__ == '__main__':
    kmeans = KMeans()
    # kmeans.initialize_centroids_random()
    # kmeans.start_clustering()
    # kmeans.plot()
    start_time = time.time()
    kmeans.initialize_centroids_plus_plus()
    kmeans.start_clustering()
    print("Time elapsed for kmeans ++: ", time.time() - start_time)
    kmeans.plot()

    start_time = time.time()
    kmeans.initialize_centroids_parallel()
    kmeans.start_clustering()
    print("Time elapsed for scalable kmeans: ", time.time() - start_time)
    kmeans.plot()
