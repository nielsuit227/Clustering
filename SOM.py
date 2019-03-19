import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

class SOM(object):
    _trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha = None, sigma = None):
        # Initializes all necessary parameters.
        # m x n = SOM Dimension
        # dim = training dim
        # alpha = learning rate
        # sigma = neighbourhood value
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n)/2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
        # Initialize Graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Initialize weightsx =
            self._weightage_vects = tf.Variable(tf.random_normal([m*n, dim]))
            # SOM Grid locations
            self._location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))),dtype=tf.int64)
            # Input dimension
            self._vect_input = tf.placeholder("float", [dim])
            self._iter_input = tf.placeholder("float")
            # Best matching unit based on Euclidean distance between weight and input
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weightage_vects, tf.stack([self._vect_input for i in range(m*n)])), 2), 1)), 0)
            # Location bmu
            slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0,1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.constant(np.array([1, 2]),dtype=tf.int64)), [2])
            # Adaptive rates
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input, self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)
            # Learning rate all neurons
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(self._location_vects, tf.stack([bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m*n)])
            weightage_delta = tf.multiply(learning_rate_multiplier, tf.subtract(tf.stack([self._vect_input for i in range(m*n)]), self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects, weightage_delta)
            self._training_op = tf.assign(self._weightage_vects, new_weightages_op)
            # Session init
            self._sess = tf.Session()
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)


    def _neuron_locations(selfs, m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])


    def train(self, input_vects):
        for iter_no in range(self._n_iterations):
            for input_vect in input_vects:
                self._sess.run(self._training_op, feed_dict={self._vect_input: input_vect, self._iter_input: iter_no})

        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
        self._trained = True


    def get_centroids(self):
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        if not self._trained:
            raise ValueError("SOM not trained yet")
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))], key = lambda x: np.linalg.norm(vect-self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return

# Setting up SOM
print('Getting SOM ready')
#SOM(Output x, Output y, input dim, iterations)
som = SOM(10,10,24,5)
# Loading data
print('Loading data')
my_data = np.load('A18.npy')
n = 100
X = my_data[:n, 3:]
m = np.size(X,1)
X = (X-np.outer(np.ones(n), np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0))

# Training SOM
print('Training SOM, might be a while...')
t = time.time()
som.train(X)
print(time.time()-t)
image_grid = som.get_centroids()
map2D = som.map_vects(X)

# Check website if you don't know how to visualize. For now doesn't make sense to implement.
# https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
