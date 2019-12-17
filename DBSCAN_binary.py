import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
fault_th = 10           # Threshold on outliers to generate fault
n = 10000               # Used samples for DBScan
ldaDimension = 2        # LDA dimension
clusterRadius = 0.5     # Cluster radius
minSamples = 10         # Minimum datapoints in cluster to be defined cluster


# Main function, loops through datapoints that are undiscovered.
def dbscan(setofpoints, eps, minpoints):
    global n
    clusterid = -1 * np.ones(n)
    for iter in range(0, n - 1):
        if clusterid[iter] == -1:
            clusterid = expandcluster(setofpoints, iter, clusterid, eps, minpoints)
    return clusterid


# Check whether a point is a center point and if so
# which points belong to the same cluster
# Checking and processing items 2-5 ms, assigning clusters 0 us.
def expandcluster(setofpoints, point, clusterid, eps, minpoints):
    global checked
    seeds = regionquery(setofpoints, point, eps)  # Returns indices
    if np.sum(seeds) < minpoints:
        return clusterid
    else:
        clusterid[seeds] = nextid(seeds)
        seeds[np.where(seeds)[0][0]] = False
        while np.sum(seeds) != 0:
            #print('Iteration %.0f, %.0f to analyze, %.0f size cluster' % (
            #    np.sum(checked), np.sum(seeds), np.sum(clusterid == clusterid[point])))
            point = np.where(seeds)[0][0]
            t = time.time()
            result = regionquery(setofpoints, point, eps)
            if np.sum(result) >= minpoints:
                clusterid[result] = clusterid[point]
                seeds = np.maximum(result, seeds)
                seeds = np.logical_and(seeds == True, checked != True)
            else:
                seeds[point] = False
    return clusterid


# Determines which data points are within cluster distance.
def regionquery(dataset, point, eps):
    global SpatialQuerySet, checked
    checked[point] = True
    neighbour = np.zeros(n) == 1
    neighbour[point] = True
    sq_point = SpatialQuerySet[point]
    dataset_sq = dataset[SpatialQuerySet == sq_point, :]
    neighbour = np.linalg.norm(dataset[point, :] - dataset_sq, 2, axis=1) < eps
    return neighbour


# Incremental cluster number
def nextid(seeds):  # Counts which cluster we're assessing
    global nextClID
    nextClID += 1
    return np.ones(np.sum(seeds)) * nextClID


# Load, split and preprocess data
print('Loading data')
# TODO filter out pumps and thermistors, ask Les about this.
my_data = np.load('A18.npy')


# Used samples and PCA dimension
X = my_data[:n, 3:]
Y = my_data[:n, [1, 2]]
m = np.size(X, 1)
X = (X-np.outer(np.ones(n), np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0))

# PCA
print('Applying PCA')
u, s, v = np.linalg.svd(X, full_matrices=False, compute_uv=True)
Xt = np.dot(X, v[:, :ldaDimension])

# # Spatial Queries
# print('Determining Spatial Queries')
# bf = 100
# nc = None
# th = 0.05
# brc = Birch(branching_factor=bf, n_clusters=nc, threshold=th, compute_labels=True)
# brc.fit(Xt)
# SpatialQuery = brc.predict(Xt)

# Define global parameters
print('Running DBSCAN')
SpatialQuerySet = np.ones(n)
nextClID = 0
checked = np.zeros(n) == 1
seeds = np.zeros(n) == 1
t = time.time()
clusters = dbscan(Xt, clusterRadius, minSamples)
print('Time for %.0f datapoints: %.3f' % (n,time.time()-t))
color = ''
plt.figure()
for i in set(clusters):
    if i == -1:
        color = 'red'
        label = 'Outlier'
    else:
        if color == 'green':
            label = ''
        else:
            label = 'Operating point'
        color = 'green'
    plt.scatter(Xt[clusters == i, 0], Xt[clusters == i, 1], c=color, label=label)
plt.show()
