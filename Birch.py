### Script which loads specific signals of Tritium's telemetry data (mainly cooling & electric measurements).
### Applies Birch, a clustering algorithm. Uses three cluster features (datapoints, sum, sum of squares).
### Generates a hierarchical cluster. 
import numpy as np
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
# Load data
my_data = np.load('A18.npy')
print('Data Loaded')
# Seperate timestamps
Y = my_data[:,[1,2]]            # 0 is left out purposely as that's just an index. Item 1 is SN, item 2 is Timestamp.
X = my_data[:,3:]               # All the rest is useable data)

################ JUST FOR NOW ##########################
n = 350000
X = X[1:n,:]

# Make zero mean and unit variance
[n,m] = np.shape(X)
X = (X-np.outer(np.mean(X,axis=1),np.ones((1,m))))/np.sqrt(np.var(X))

## PCA
u,s,v = np.linalg.svd(X,full_matrices=False,compute_uv=True)
Xt = np.dot(X,v[:,:2])

# Birch settings
bf = 100                # Branch factor, maximum nodes in a branching_factor
nc = None               # Maximum clusters of output
th = 0.1               # Maximum cluster radius


# Settings and Training of Birch
print('Birch training')
t = time.time()
brc = Birch(branching_factor=bf, n_clusters=nc, threshold=th, compute_labels=True)
brc.fit(Xt)
elapsed = time.time()-t
print('Training done, elapsed time %.2f' % elapsed)


Yp = brc.predict(Xt)
Y = brc.transform(Xt)
[n,c] = np.shape(Y)
print(str(c)+'  classes assigned')
for i in range(0,c):
        print('Class '+str(i)+' | '+str(sum(Yp==i)))

fig = plt.figure()
X2 = np.dot(X,v[:,:2])
plt.scatter(X2[:,0],X2[:,1],c=Yp)
plt.show()
