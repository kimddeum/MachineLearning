import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
# import scipy.misc
import matplotlib.pyplot as plt
import random
import copy

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)

    copied from http://deeplearning.net/ and revised by hchoi
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    return train_set, valid_set, test_set


class kMeans():

    def __init__(self, k, dim):
        self.k = k
        self.dim = dim

    def init_centroids(self):
        print('\t Initializing centroids ...')

        centroids = np.zeros((self.k, self.data.shape[1]))
        for i in range(0, self.k):
            idx = random.randint(0, self.data.shape[0])
            centroids[i, :] = self.data[idx]

        return centroids  # 2d array with centroid data.

    def form_clusters(self, centroids):
        print('\t Making clusters')

        clusters = {c: [] for c in range(len(centroids))}
        for i in np.ndindex(self.data.shape[0]):  # check all the data / i is a tuple
            dist = np.linalg.norm(self.data[i] - centroids, axis=1)
            clusters[np.argmin(dist)].append(int(i[0]))

        return clusters  # dictionary form

    def update_centroids(self, clusters):
        print('\t Updating centroids')
        new_centroids = np.zeros((self.k, self.data.shape[1]))

        for cent_id, items in clusters.items():
            new = np.mean(self.data[items], axis=0)
            new_centroids[cent_id] = new

        if dim is None:
            return new_centroids
        else:
            return new_centroids.reshape(len(new_centroids), -1)

    def fit(self, data):
        self.data = data
        print('\t Starting kMeans algorithm')
        prev_diff = 0
        centroids = self.init_centroids()
        clusters = self.form_clusters(centroids)

        while True:
            print('\t\t Updating centroids')
            old_centroids = centroids
            centroids = self.update_centroids(clusters)
            clusters = self.form_clusters(centroids)

            diff = map(lambda a, b: np.linalg.norm(a - b), old_centroids, centroids)
            max_diff = max(diff)

            changes = abs(max_diff - prev_diff)
            prev_diff = max_diff

            if changes < 10e-6:
                break

        return centroids, clusters


def extract_feature(feature, train_set):
    print('\tExtracting data where label is ', feature)

    data, label = train_set

    index = list(np.where(np.isin(label, feature))[0])
    extracted_data = np.zeros((len(index), len(data[0])))

    for i, id in enumerate(index):
        extracted_data[i] = data[id]

    return extracted_data


def eigenprojection(data, dim=2):

    cov = np.cov(data.T)
    _, eigenvectors = np.linalg.eig(cov)
    eigenvectors = eigenvectors.real.T[:dim].T

    eigen_data = np.dot(data, eigenvectors)

    return eigen_data


def draw_scatter(k, dim, data, centroids, clusters):

    fig, ax = plt.subplots()

    for i, j in clusters.items():
        clustered_data = np.array(data[j])

        if dim is None:
            clustered_data = eigenprojection(clustered_data)
            centroids = eigenprojection(centroids)
        ax.scatter(clustered_data[:, 0], clustered_data[:, 1], s=7, cmap=plt.cm.get_cmap('rainbow'))

    ax.scatter(centroids[:, 0], centroids[:, 1], s=20, marker='x', cmap='black')
    fig.savefig('./Result/k={}_dim={}.png'.format(k, dim))


if __name__ == '__main__':
    train_set, _, _ = load_data('mnist.pkl.gz')

    feature = [3, 9]
    num_cluster = [2, 3, 5, 10]
    eigen_dim_list = [None, 2, 5, 10]

    data = extract_feature(feature, train_set)

    for dim in eigen_dim_list:
        for k in num_cluster:

            if dim is not None:
                data = eigenprojection(data, dim)

            centroids, clusters = kMeans(k, dim).fit(data)
            draw_scatter(k, dim, data, centroids, clusters)
