import numpy as np, time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import hdbscan
import warnings
warnings.filterwarnings("ignore")

### Parameters
fault_th = 10                   # Threshold on outliers to generate fault
split = 10                      # Split data in split sections
ldaDimension = 2                # LDA dimension
clusterRadius = 1               # Cluster radius
minSamples = 25                 # Minimum datapoints in cluster to be defined cluster
trainSamples = 100000           # Used samples for clustering

# Load, split and preprocess data
str = 'Sep18'
print('Loading '+str+' dataset')
my_data = np.load(str+'.npy')
# Used samples and PCA dimension
X = my_data[:, 3:]
Y = my_data[:, [0, 1]]
n, m = np.shape(X)
X = (X-np.outer(np.ones(n), np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0))

# PCA
print('Calculating Singular Value Decomposition for Dimension Reduction')
u, s, v = np.linalg.svd(X, full_matrices=False, compute_uv=True)
X = np.dot(X, v[:, :ldaDimension])

# Select samples
idx = np.random.permutation(n)
Xt = X[idx[:trainSamples], :]
n, m = np.shape(Xt)

plt.figure()
plt.scatter(Xt[:, 0], Xt[:, 1])
plt.show()

# DBSCAN
print('Evaluating Euclidean distance to determine clusters')
print('Cluster Radius   = %.2f' % clusterRadius)
print('Minimum point    = %.0f' % minSamples)
t = time.time()
clusterer = hdbscan.hdbscan(Xt, min_cluster_size=minSamples, gen_min_span_tree=True)
elapsed = time.time() - t
print('Training done; Elapsed time %.2f seconds\n' % elapsed)

clusters = clusterer[0]
confidence = clusterer[1]
clusterSize = clusterer[2]
tree = clusterer[3]

# Plot 2d - selected data
print('Plotting results')
outlier = clusters == -1
plt.scatter(Xt[:, 0], Xt[:, 1], c=clusters)
plt.scatter(Xt[outlier, 0], Xt[outlier, 1], c='r')
plt.show()
