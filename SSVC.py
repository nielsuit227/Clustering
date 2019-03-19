import numpy as np, time
import matplotlib.pyplot as plt
import sklearn.datasets
plt.style.use('seaborn-whitegrid')
from scalable_support_vector_clustering import ScalableSupportVectorClustering

# Parameters
supportVectors = 100
kernelWidth = 20                   # Inverse
outlierFraction = 0.00005
epochs = 2

# Load data
my_data = np.load('Sep18.npy')
n = 100000
X = my_data[:n, 3:]
m = np.size(X, 1)
X = (X-np.outer(np.ones(n), np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0))

# PCA
dim = 2
u, s, v = np.linalg.svd(X, full_matrices=False, compute_uv=True)
X = np.dot(X, v[:, :dim])
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# Scalable Support Vector Classification
t = time.time()
ssvc = ScalableSupportVectorClustering()
ssvc.dataset(X)
ssvc.parameters(p=outlierFraction, B=supportVectors, q=kernelWidth, eps1=0.15, eps2=10**-5, step_size=10**-1)
ssvc.find_alpha(epochs=epochs)
ssvc.cluster()
ssvc.show_plot(figsize=(8, 6))
#ssvc.show_bdd(minx=-2.1, maxx=5, miny=-3.5, maxy=1.5, n=50, figsize=(8, 6))
print('Training time for %.0f samples: %.2f' % (n, time.time()-t))