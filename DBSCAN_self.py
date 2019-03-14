import numpy as np
import matplotlib.pyplot as plt
import time
## Reckon that binary vector for checked, seeds and results are computationally easier to compute




# Parameters
fault_th = 10  # Threshold on outliers to generate fault
n = 2000  # Used samples for DBScan
ldaDimension = 2  # LDA dimension
clusterRadius = 0.05  # Cluster radius
minSamples = 10  # Minimum datapoints in cluster to be defined cluster


# Main function, loops through datapoints that are undiscovered.
def dbscan(setofpoints, eps, minpoints):
    global n
    clusterid = -1 * np.ones(n)
    for i in range(0, n-1):
        if clusterid[i] == -1:
            clusterid = expandcluster(setofpoints, i, clusterid, eps, minpoints)
    return clusterid


# Check whether a point is a center point and if so
# which points belong to the same cluster
# CHECK WHETHER SPEEDING UP ISN'T POSSIBLE (LIN ALG)
def expandcluster(setofpoints, point, clusterid, eps, minpoints):
    global checked
    seeds = regionquery(setofpoints, point, eps)  # Returns indices
    if len(seeds) < minpoints:
        return clusterid
    else:
        clusterid[seeds] = nextid(seeds)
        seeds = np.delete(seeds, 0)
        while len(seeds) != 0:
            print('Iteration %.0f, %.0f to analyze, %.0f size cluster' % (len(checked), len(seeds), sum(clusterid==clusterid[point])))
            point = seeds[0]
            result = regionquery(setofpoints, point, eps)
            if len(result) >= minpoints:
                clusterid[result] = clusterid[point]
                seeds = np.unique(np.append(seeds, result))
                for i in checked:
                    ind = np.where(seeds == i)
                    seeds = np.delete(seeds, ind)
            else:
                seeds = np.delete(seeds, 0)
    return clusterid


# Determines which data points are within cluster distance.
# SPEED UP CHECK
def regionquery(dataset, point, eps):
    global SpatialQuerySet, checked  # Make sure size = (10,)
    checked = np.append(checked, point)
    seeds = [point]
    sq_point = SpatialQuerySet[point]
    dataset_sq = dataset[SpatialQuerySet == sq_point, :]
    for i in range(0, len(dataset_sq[:, 1])):
        if np.linalg.norm(dataset[point, :] - dataset_sq[i, :], 2) < eps:
            seeds = np.append(seeds, i)
            seeds = np.unique(seeds)
    return seeds


# Incremental cluster number
def nextid(seeds):  # Counts which cluster we're assessing
    global nextClID
    nextClID += 1
    return np.ones(len(seeds)) * nextClID


# Load, split and preprocess data
print('Loading data')
my_data = np.load('A18.npy')

# Used samples and PCA dimension
X = my_data[:n, 3:]
Y = my_data[:n, [1, 2]]
m = np.size(X, 1)
X = (X - np.outer(np.mean(X, axis=1), np.ones((1, m)))) / np.sqrt(np.var(X))

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
print('DBSCAN')
SpatialQuerySet = np.ones(n)        # OBVIOUSLY FUCKED
nextClID = 0
checked = []
t = time.time()
clusters = dbscan(Xt,clusterRadius,minSamples)
print('Time for %.0f datapoints: %.1f' % (n,(time.time()-t)))


plt.figure()
plt.scatter(Xt[:,0],Xt[:,1],c=clusters)
plt.show()