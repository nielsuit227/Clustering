import numpy as np
from numpy.random import random_integers as rdmint
import matplotlib.pyplot as plt
import time
from matplotlib.textpath import TextPath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from pandas.plotting import parallel_coordinates
import pandas as pd

def kd_tree(data, kdim=5, leafsize=10, random=None):
    if random is None:
        random = True
    ndata, ndim = np.shape(data)
    if random:
        sdim = np.arange(kdim)
        ind = sdim[rdmint(0, kdim - 1)]
    else:
        ind = 0
    idx = np.argsort(data[:, ind], kind='quicksort')
    data = data[idx, :]
    # Add split
    stack = []
    # Tree: Data, Indeces, Split Dim., Split Val.
    # Stack: Data, Indices, Node #, Leaf bool, Depth
    tree = [(None, idx, 0, ind, data[int(ndata / 2), ind])]
    stack.append((data[:int(ndata / 2), :], idx[:int(ndata / 2)], 1, int(ndata / 2) < leafsize, 0))
    stack.append((data[int(ndata / 2):, :], idx[int(ndata / 2):], 2, int(ndata / 2) < leafsize, 0))
    # Iteratively loop through stack (to do list)
    while stack:
        data, idx, node, leaf, depth = stack.pop()
        if leaf:
            tree.append((data, idx, node, None, None))
        else:
            if random:
                ind = sdim[rdmint(0, kdim - 1)]
            else:
                ind = np.remainder(depth+1, ndim)
            tidx = np.argsort(data[:, ind], kind='quicksort')
            data = data[tidx, :]
            idx = idx[tidx]
            ndata, ndim = np.shape(data)
            stack.append((data[:int(ndata / 2), :], idx[:int(ndata / 2)], node * 2 + 1, int(ndata / 2) < leafsize,
                          depth+1))
            stack.append((data[int(ndata / 2):, :], idx[int(ndata / 2):], node * 2 + 2, int(ndata / 2) < leafsize,
                          depth+1))
            tree.append((None, idx, node, ind, data[int(ndata / 2), ind]))
    return tree


def sortnode(val):
    return val[2]


def radius_nn(tree, point, radius):
    stack = [tree[0]]
    while stack:
        data, idx, node, sdim, sval = stack.pop()
        if data is not None:
            distance = np.sqrt(np.sum((data - point) ** 2, 1))
            return idx[np.where(distance < radius)]
        else:
            if point[sdim] > sval:                      # Make beq to isolate leaves
                stack.append(tree[node * 2 + 2])
            else:
                stack.append(tree[node * 2 + 1])


def visualizetwo(data, tree):
    xmi = min(1.1*np.min(data[:, 0]), 0.9*np.min(data[:, 0]))
    xma = 1.1*np.max(data[:, 0])
    ymi = min(1.1*np.min(data[:, 1]), 0.9*np.min(data[:, 1]))
    yma = 1.1*np.max(data[:, 1])
    hrect = [(0, xmi, xma, ymi, yma)]
    store = []
    copy = data.copy()
    plt.scatter(data[:, 0], data[:, 1], c='k', s=1)
    while hrect:
        plt.scatter(copy[:, 0], copy[:, 1], c='k', s=1)
        parent, pxmi, pxma, pymi, pyma = hrect.pop()
        data, idx, node, sdim, sval = tree[parent]
        if data is not None:
            continue
        if sdim == 1:
            hrect.append((2*parent+2, pxmi, pxma, sval, pyma))
            hrect.append((2*parent+1, pxmi, pxma, pymi, sval))
            plt.plot([pxmi, pxma], [sval, sval])
            store.append((pxmi, pxma, sval, sval))
        else:
            hrect.append((2*parent+1, pxmi, sval, pymi, pyma))
            hrect.append((2*parent+2, sval, pxma, pymi, pyma))
            plt.plot([sval, sval], [pymi, pyma])
            store.append((sval, sval, pymi, pyma))
        for i in range(len(store)):
            xmin, xmax, ymin, ymax = store[i]
            plt.plot([xmin, xmax], [ymin, ymax])
        plt.show()


def visualize(data, tree, label=None, y=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    xmi = min(1.5*np.min(data[:, 0]), 0.5*np.min(data[:, 0]))
    xma = 1.5*np.max(data[:, 0])
    ymi = min(1.5*np.min(data[:, 1]), 0.5*np.min(data[:, 1]))
    yma = 1.5*np.max(data[:, 1])
    hrect = [(0, xmi, xma, ymi, yma)]
    if any(label):
        plt.scatter(data[:, 0], data[:, 1], c='g', s=1)
        plt.scatter(data[label, 0], data[label, 1], c='r', s=1)
        for olind in np.where(label == 1)[0]:
            fp = FontProperties(family="Courier")
            tp1 = TextPath((data[olind, 0], data[olind, 1]), str(int(y[olind])), size=0.05, prop=fp)
            polygon = tp1.to_polygons()
            for a in polygon:
                p1 = patches.Polygon(a, fill=False)
                ax1.add_patch(p1)
    else:
        plt.scatter(data[:, 0], data[:, 1], c='k', s=1)
    while hrect:
        parent, pxmi, pxma, pymi, pyma = hrect.pop()
        data, idx, node, sdim, sval = tree[parent]
        if data is not None:
            continue
        if sdim == 1:
            hrect.append((2*parent+2, pxmi, pxma, sval, pyma))
            hrect.append((2*parent+1, pxmi, pxma, pymi, sval))
            plt.plot([pxmi, pxma], [sval, sval])
        else:
            hrect.append((2*parent+1, pxmi, sval, pymi, pyma))
            hrect.append((2*parent+2, sval, pxma, pymi, pyma))
            plt.plot([sval, sval], [pymi, pyma])


class DBSCAN(object):

    def __init__(self, data, radius=1.0, minpoints=50, tree=None, kdim=None, leafsize=None):
        if tree is not None:
            self._tree = tree
        else:
            if kdim is not None and leafsize is not None:
                self._tree = rkd_tree(data, kdim=kdim, leafsize=leafsize)
            else:
                print('Assign either tree, or kdim & leafsize')
        self._data = data
        self._n, self._m = np.shape(self._data)
        self._eps = radius
        self._minpoints = minpoints
        self._clusterid = -1 * np.ones(self._n)
        self._checked = False * np.ones(self._n)
        self.next_cluster_id = 0

    def fit(self):
        self._n, self._m = np.shape(self._data)
        for i, point in enumerate(self._data):
            if self._clusterid[i] == -1:
                self.expandcluster(point=i)
        return self._clusterid

    # todo, speedup by checking leafe size. High prob all are clusterpoints.
    def expandcluster(self, point):
        seeds = self.withinrad(point)
        if np.sum(seeds) < self._minpoints:
            return self._clusterid
        else:
            self._clusterid[seeds] = self.nextid(seeds)
            seeds[np.where(seeds)[0][0]] = False
            while np.sum(seeds) != 0:
                print('[%.2f %%]Iteration %.0f, %.0f to analyze, %.0f size cluster' % (np.sum(self._checked)*100/self._n,
                      np.sum(self._checked), np.sum(seeds), np.sum(self._clusterid == self._clusterid[point])))
                npoint = np.where(seeds)[0][0]
                nseeds = self.withinrad(npoint)
                if np.sum(nseeds) >= self._minpoints:
                    self._clusterid[nseeds] = self._clusterid[npoint]
                    seeds = np.maximum(nseeds, seeds)
                    seeds = np.logical_and(seeds == True, self._checked == False)
                else:
                    seeds[npoint] = False
        return

    def withinrad(self, point):
        self._checked[point] = True
        seeds = np.zeros(self._n, dtype='bool')
        seedsindex = radius_nn(self._tree, point=self._data[point, :], radius=self._eps)
        seeds[seedsindex] = True
        return seeds

    def nextid(self, seeds):
        self.next_cluster_id += 1
        return np.ones(int(np.sum(seeds))) * self.next_cluster_id


# Import & process data
n = 500000
X = np.load('Jan18.npy')
Y = X[:n, 0:2]
X = X[:n, 3:]
# flags: 1-3, soc 22, Ws 20,23,5,9
dlt = []#[1, 2, 3, 5, 9, 20, 22, 23]
X = np.delete(X, dlt, 1)
n, m = np.shape(X)
X = (X - np.outer(np.ones(n), np.mean(X, axis=0))) / np.sqrt(np.var(X, axis=0))
u, s, v = np.linalg.svd(X, full_matrices=False, compute_uv=True)
X = np.dot(X, v[:, :])
n, m = np.shape(X)

# Set up tree
t = time.time()
Tree = kd_tree(X, kdim=2, leafsize=250, random=False)
Tree.sort(key=sortnode)
buildtime = time.time() - t
print('Building tree took %.3f s, (size = %.0f)' % (buildtime, len(Tree)))

# Set up dbscan
t = time.time()
cluster = DBSCAN(X, radius=6, minpoints=10, tree=Tree)
cluster.fit()
dbscantime = time.time() - t
print('Clustering points took %.3f s' % dbscantime)
OLbool = cluster._clusterid == -1

# Plot result
if m == 2:
    visualize(X, Tree, OLbool, Y[:, 0])
else:
    plt.figure()
    for i in range(n):
        if OLbool[i] == True:
            plt.plot(X[i, :], c='r')
    n = 1000
    sX = X[:n, :]
    sOLbool = OLbool[:n]
    for i in range(n):
        if sOLbool[i] == False:
            plt.plot(sX[i, :], c='g')
    plt.suptitle('Parrallel Coordinates plot')
print('All done (%.0f samples)' % n)
print('Outliers in January:\n')
Z = np.zeros((len(set(Y[OLbool, 0])), 2))
for i, charger in enumerate(set(Y[OLbool, 0])):
    Z[i, :] = [charger, np.sum(Y[OLbool, 0] == charger)]
idx = np.argsort(-Z[:, 1])
Z = Z[idx, :]
for item in Z:
    print('SN: %.0f, %.0f outliers' % (item[0], item[1]))
plt.show()
