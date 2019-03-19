from typing import Any, Union

import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
fault_th = 10               # Threshold on outliers to generate fault
n = 20000                   # Used samples for DBScan
ldaDimension = 2            # LDA dimension
clusterRadius = 0.05        # Cluster radius
minSamples = 1000           # Minimum datapoints in cluster to be defined cluster

# Density Based Spatial Clustering of Applications with Noise (DBSCAN).
# Specific implementation preorganizes the dataset by generating a grid for spatial indexing.
# Specific implementation only separates noise points from clusters. In order to implement the algo with different
# clusters, a cluster counter should be assigned instead of a simple 1.
class DBSCAN(object):

    # Initializes the DBSCAN class. Epsilon is cluster radius, min_points is cluster definition, input space is n x m.
    def __init__(self, n, m, eps=None, min_points=None):
        self._n = n
        self._m = m
        if eps is None:
            eps = 0.1
        if min_points is None:
            min_points = 10
        self._eps = eps
        self._min_points = min_points
        self._cell_width = self._eps / np.sqrt(2)
        self._cluster = -1 * np.ones(n)
        self._cluster_count = 1
        self._tol = 1e-4

    # TODO make multi-dimensional (now just 2D)
    # TODO implement linear algrebra to speed up
    # Function that generates a grid based on the input interval and cluster radius. It ensures that all points
    # within one cell are also in the same cluster. The output is a vector, self._grid (in R^n), with the index of the
    # cells belonging to the datapoints. It is counted from left to right from top to bottom.
    def grid(self, Xt):
        x_min = np.amin(Xt[:, 0])
        y_min = np.amin(Xt[:, 1])
        self._n_rows = np.floor((np.amax(Xt[:, 0]) - x_min) / self._cell_width + 1).astype(int)
        self._n_cols = np.floor((np.amax(Xt[:, 1]) - y_min) / self._cell_width + 1).astype(int)
        self._grid = np.zeros(n)
        for i, point in enumerate(Xt):
            indx = np.floor((point[0] - x_min) / self._cell_width + 1).astype(int)
            indy = np.floor((point[1] - y_min) / self._cell_width + 1).astype(int)
            self._grid[i] = (indx - 1) * self._n_cols + indy

    # TODO figure out why we need to know the grid centers
    # TODO implement linear algebra to speed up
    # Generates the centerpoints of each cell belonging to the grid. Necessary to assess which cells are neighbours.
    def centers(self):
        self._grid_center = np.zeros([self._n_rows*self._n_cols, self._m])
        for i in range(1, self._n_cols*self._n_rows):
            idx = np.ceil(i/self._n_cols)
            idy = np.remainder(i, self._n_cols)
            self._grid_center[i, :] = [(0.5 + idx) * self._cell_width, (0.5 + idy) * self._cell_width]

    # TODO figure out whether this works
    def find_core(self):
        for i, cell in enumerate(self._grid):
            if self._cluster[i] == -1:
                if np.sum(self._grid == cell) >= self._min_points:
                    self._cluster[np.where(self._grid == cell)] = 1
                else:
                    in_current = np.where(self._grid == cell)
                    boolean_neighbours = np.linalg.norm(self._grid_center[cell.astype(int)] - self._grid_center,
                                                        2, axis=1) <= self._eps + self._tol
                    neighbour_grid = np.where(boolean_neighbours)
                    in_neighbour = np.zeros(n)
                    for neighbour_cell in neighbour_grid[0]:
                        in_neighbour[np.where(self._grid == neighbour_cell)] = 1
                    for j, point_cell in enumerate(X[in_current, :][0]):
                        n_points = -1               # -1 to avoid removing itself, which is obviously in neighbourhood
                        for point_neighbour in X[np.where(in_neighbour == 1), :][0]:
                            if np.linalg.norm(point_cell - point_neighbour, 2) < self._eps:
                                n_points += 1
                                if n_points >= self._min_points:
                                    self._cluster[in_current[0][j]] = 1

    # TODO implement merging clusters. Crossref all corepoints to corepoints of OTHER clusters.
    def merge_clusters(self):
        core_points = np.where(self._cluster != -1)
        for core in core_points:
            other_cluster_cores = np.where(np.logical_and(self._cluster != -1, self._cluster != self._cluster[core]))
            for other in other_cluster_cores:
                if np.linalg.norm(core - other, 2) < self._eps:
                    self.cluster

    #def find_outliers(self):



    # def predict(self,X):


# Load, split and preprocess data
print('Loading data')
my_data = np.load('A18.npy')

# Used samples and PCA dimension
X = my_data[:n, 3:]
Y = my_data[:n, [1, 2]]
m = np.size(X, 1)
X = (X-np.outer(np.ones(n), np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0))

# PCA
u, s, v = np.linalg.svd(X, full_matrices=False, compute_uv=True)
X = np.dot(X, v[:, :ldaDimension])
n, m = np.shape(X)

dbs = DBSCAN(n, m, eps=clusterRadius, min_points=minSamples)
dbs.grid(X)
dbs.centers()
dbs.find_core()
dbs.find_outliers()

