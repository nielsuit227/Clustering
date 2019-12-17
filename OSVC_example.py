import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons as data
from self.OSVC import OSVC

# Parameters
train = 1                       # Boolean whether to load or train new
real_data = 1                   # Boolean to load either real or toy data
data_limit = 1000               # Amount of data points analysed in SVC
Multipliers = 50                # Starting amount of multipliers
kernelWidth = 0.1              # SVC Gaussian kernel variance
learningRate = 0.7              # Learning rate of OGD (for SVC)
maxIntPDF = 9000000             # Maximum sum of multipliers
memorySize = 1000               # Maximum memory before updating data representation (GMM)
outlierThreshold = 0.7          # Threshold to decide whether new point is properly reprsented
selectionSize = 10000           # Data size for initial GMM
deleteCols = [5, 9, 20, 22, 23]        # Remove SoC, Ws & flags

# Load & Preprocess data
X = data(n_samples=data_limit, shuffle=True, noise=0.01)[0]
n, m = np.shape(X)
X = (X - np.outer(np.ones(n), np.mean(X, axis=0))) / np.sqrt(np.var(X, axis=0))  # Normalize data

# OSVC
cluster = OSVC(multipliers=Multipliers, kernel_width=kernelWidth, learning_rate=learningRate, max_int_pdf=maxIntPDF,
               memory_size=memorySize, outlier_threshold=outlierThreshold, selection_size=selectionSize)
cluster._gmm(X)
mema = np.zeros(data_limit+1)
epoch = 1
for j in range(0, epoch):
    for i, point in enumerate(X):
        XC, alpha, k, outliers = cluster.update(point)
        mema[i] = alpha[5]
        print('[%.2f %%] OSVC training - %.0f multipliers - %.0f representation outliers - iteration %.0f' %
              (100 * (i + 1 + j * data_limit) / data_limit / epoch, k, outliers - k, i + 1 + j * data_limit))
print('\n\n*** Training done ***\n\n')
plt.figure()
plt.plot(mema)
plt.suptitle('Convergence of multiplier')

# Visualization
print('Working on visualization')
if train:
    if np.size(X, axis=1) == 2:
        cluster.visualize(X, grid_size=100)
plt.show()
