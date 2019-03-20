# Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as skGMM
from matplotlib.patches import Ellipse
from sklearn.datasets import make_moons as data


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=50 * w * w_factor)
    plt.suptitle('Gaussian Mixture Model - Data representation')


class OSVC(object):

    def __init__(self, X, multipliers=100, kernel_width=2.0, learning_rate=0.001, max_int_pdf=100.0, memory_size=5000, outlier_threshold=0.2,
                 gmm_var=np.eye(2), selection_size=5000):
        self._memorySize = memory_size
        self._outlierThreshold = outlier_threshold
        self._kernelWidth = kernel_width
        self._learningRate = learning_rate
        self._c = max_int_pdf
        self._GMM_variance = gmm_var
        self._selectionData = selection_size
        self._mp = multipliers
        self._alpha = np.zeros(self._mp)
        print('Gaussian Mixture Model for data representation, %.0f to %.0f samples' % (self._selectionData, self._mp))
        X_sel = X[:self._selectionData, :]
        self._gmm = skGMM(n_components=self._mp, covariance_type='diag').fit(X_sel)                     # Set up SciKit GMM & train
        self._xc = self._gmm.means_                                             # Select means
        self._outlierMemory = self._xc                                          # Store in memory
        if np.size(X,axis=1) == 2:
            plot_gmm(self._gmm, X_sel)                                          # Show what it did.

    def SVC_update(self, xt):
        grad = self._kernel(self._xc, xt)                               # Gradient
        self._alpha = self._alpha + self._learningRate * grad           # Gradient Descent step
        if np.sum(self._alpha) >= self._c:                              # If alpha's too high
            self._alpha = self._alpha / np.sum(self._alpha) * self._c   # Projecting for constraint (sum kernel < C)
        # Check whether GMM needs an update
        p = self._gmm.predict_proba(np.reshape(xt, (1, -1)))            # Calculate prob. xt belongs to xc
        if ~(p >= self._outlierThreshold).any():                           # If P(xt ~in xc) < thres, save it
            self._outlierMemory = np.vstack((self._outlierMemory, xt))
            if np.size(self._outlierMemory, axis=0) >= self._memorySize:
                print('Refitting data representation, one moment s.v.p.')
                self._mp += 50
                self._memorySize += 100
                self._gmm = skGMM(n_components=self._mp, covariance_type='diag', means_init=self._xc)
                self._gmm.fit(self._outlierMemory)
                self._xc = self._gmm.means_
                self._outlierMemory = self._xc
        return self._xc, self._alpha, np.size(self._outlierMemory, axis=0)

    def _kernel(self, x, y):
        if x.ndim == 1:
            return np.exp(-np.dot(x - y, x - y) / 2 / self._kernelWidth / self._kernelWidth)
        else:
            return np.exp(-np.diag(np.dot((x - y), (x - y).T)) / 2 / self._kernelWidth / self._kernelWidth)

    def visualize(self, data, grid_size=100):
        xmi = 1.5*np.min(self._xc)
        xma = 1.5*np.max(self._xc)
        grid = np.linspace(xmi, xma, grid_size)
        f = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                for k, alpha in enumerate(self._alpha):
                    f[j, i] = f[j, i] + alpha*self._kernel(np.array([grid[i], grid[j]]), XC[k, :])
        plt.figure()
        plt.contourf(grid, grid, f)
        plt.scatter(data[:, 0], data[:, 1], s=1)
        plt.scatter(self._xc[:, 0], self._xc[:, 1], s=5, c='k')
        plt.suptitle('Density approximation plus data points (blue) and represnetation (black)')

    def SVC_predict(self, xp):
        f = np.dot(self._alpha, self._kernel(self._xc, xp))
        if f > self._outlierThreshold:
            return 1
        else:
            return 0
    # TODO implement SVC prediction


# Parameters
data_limit = 100000  # Amount of data points analysed in SVC

# # Load & preprocess data
real_data = 1;
if real_data:
    my_data = np.load('Sep18.npy')  # Load data
    X = my_data[:data_limit, 3:]  # Select sensor data
    Y = my_data[:data_limit, [0, 1]]  # Select serial and timestamp
    n, m = np.shape(X)  # Note size
    X = (X - np.outer(np.ones(n), np.mean(X, axis=0))) / np.sqrt(np.var(X, axis=0))  # Normalize data
    idx = np.random.permutation(n)  # Get shuffle indices
    X = X[idx, :]  # Shuffle sensor data
    Y = Y[idx, :]  # Match labels

    # Principle Component Analysis - Dimension Reduction
    PCA_reductionDimension = 2  # Dimension of data used for GMM, SVC
    print('Principle Component Analysis for dimension reduction, to %.0f dimensions' % PCA_reductionDimension)
    u, s, v = np.linalg.svd(X, full_matrices=False, compute_uv=True)  # Singular Value Decomposition (for PCs)
    X = np.dot(X, v[:, :PCA_reductionDimension])  # Matrix transformation with dim. limit
else:
    X = data(n_samples=data_limit, shuffle=True, noise=0.01)[0]
    n,m = np.shape(X)
    X = (X - np.outer(np.ones(n), np.mean(X, axis=0))) / np.sqrt(np.var(X, axis=0))  # Normalize data


# Online Support Vector Clustering (Gaussian Mixture Model inside)
k = 500             # Multipliers
cluster = OSVC(X, multipliers=k, kernel_width=0.1, learning_rate=0.0075, max_int_pdf=500, memory_size=1000,
               outlier_threshold=0.5, gmm_var=np.eye(2), selection_size=10000)
mema = np.zeros(data_limit)
for i, point in enumerate(X):
    XC, alpha, outliers = cluster.SVC_update(point)
    mema[i] = alpha[5]
    print('[%.2f %%] OSVM training iteration %.0f - %.0f representation outliers' %(100*(i+1)/data_limit, i+1, outliers-k))
print('\n\n*** Training done, working on visualization ***\n\n')
if np.size(X, axis=1) == 2:
    cluster.visualize(X, grid_size=100)
    plt.figure()
    plt.plot(mema)
    plt.suptitle('Convergence of multipliers')
    plt.show()
print('Resulting multipliers (saved)')
print(alpha)
np.save('Last_OSVM_result.npy',alpha,XC)



