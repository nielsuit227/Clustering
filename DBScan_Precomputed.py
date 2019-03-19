import numpy as np, time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

### Parameters
fault_th = 10                   # Threshold on outliers to generate fault
split = 10                      # Split data in split sections
ldaDimension = 2                # LDA dimension
clusterRadius = 0.5             # Cluster radius
minSamples = 100                # Minimum datapoints in cluster to be defined cluster
n = 10000                       # Used samples

# Load, split and preprocess data
print('Loading data')
my_data = np.load('A18.npy')
# Used samples and PCA dimension
X = my_data[:n,3:]
Y = my_data[:n,[1,2]]
[n,m] = np.shape(X)
X = (X-np.outer(np.ones(n), np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0))

# PCA
print('Applying PCA')
u,s,v = np.linalg.svd(X,full_matrices=False,compute_uv=True)
Xt = np.dot(X,v[:,:ldaDimension])
n,m = np.shape(Xt)
# DBSCAN
print('Precomputing distance matrix')
t = time.time()
neigh = NearestNeighbors(radius = clusterRadius)
neigh.fit(Xt)
dist = neigh.radius_neighbors_graph(mode='distance')
print('Clustering with DBSCAN')
dbs = DBSCAN(eps=clusterRadius,min_samples = minSamples, metric='precomputed').fit(dist)
elapsed = time.time() - t
print('Training done; Elapsed time %.2f seconds\n' % elapsed)

# Plot 2d
color = None
if m == 2:
    fig = plt.figure()
    for i in set(dbs.labels_):
        if i == -1:
            color = 'red'
            label = 'Outlier'
        else:
            if color == 'green':
                label = ''
            else:
                label = 'Operating point'
            color = 'green'

        plt.scatter(Xt[dbs.labels_==i,0],Xt[dbs.labels_==i,1],c=color,label=label)
        plt.draw()
    fig.suptitle('Operating point and outliers')
    fig.legend()
print()
# Print outliers
outlierIndex = np.where(dbs.labels_ == -1)
for i in outlierIndex:
    outlierIndex = i
outlierSerial = Y[outlierIndex,0]
outlierTime = Y[outlierIndex,1]
for i in outlierIndex:
    print('Outlier detected: Serial %.0f,    Time:' % Y[i,0] + time.strftime("%a, %d %b %Y %H:%M:%S  GMT+10", time.localtime(Y[i,1])) + '\n')
# Check for multiples
n_faults = 0
for i in set(outlierSerial):
    Z = np.sum(outlierSerial==i)
    print('Charger %.0f has %.0f outliers' %(i, Z))
    if Z >= fault_th:
        n_faults += 1
n_clusters = len(set(dbs.labels_))-(1 if -1 in dbs.labels_ else 0)
n_outliers = list(dbs.labels_).count(-1)
n_chargers = len(set(outlierSerial))
print('\n')
print('Training time                            %.3f' % elapsed)
print('Estimated number of clusters             %.0f' % n_clusters)
print('Estimated number of outliers             %.0f' % n_outliers)
print('Estimated chargers with outliers         %.0f' % n_chargers)
print('Chargers with more than %.0f outliers      %.0f' % (fault_th, n_faults))
print('\n\n')
print('Z-Score' % metrics.calinski_harabaz_score(Xt, dbs.labels_))
plt.show()
