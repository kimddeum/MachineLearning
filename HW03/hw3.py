import gzip
import os

import six.moves.cPickle as pickle
# import scipy.misc
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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


def kNN(train_x, train_y, test_x, test_y, k):
    accuracy = 0.0

    for i in range(len(test_x)):
        distance = np.linalg.norm(test_x[i] - train_x, axis=1)
        sortDist = np.argsort(distance)

        if test_y[i] == mode(train_y[sortDist[:k]])[0][0]:
            accuracy += 1.0 / (len(test_x))

    return accuracy


def eigenprojection(train_x,test_x, dim):

    cov = np.cov(train_x.T)
    _, eigenvectors = np.linalg.eig(cov)
    eigenvectors = eigenvectors.T[:dim].T

    eigen_train = np.dot(train_x, eigenvectors)
    eigen_test = np.dot(test_x, eigenvectors)

    return eigen_train, eigen_test


def random_forest(trees, depth, train_x, train_y, test_x, test_y):
    rf = RandomForestClassifier(n_estimators=trees, max_depth=depth).fit(train_x, train_y)

    return rf.score(test_x, test_y)


if __name__ == '__main__':
    train_set, _, test_set = load_data('mnist.pkl.gz')

    # kNN for eigen projected data on 2, 5, and 10 dimensions

    print("kNN for 2, 5, 10 eigen dimension")

    eigen_dim_list = [None, 2, 5, 10]
    neighbor_list = [1, 5, 10]

    for eigen_dim in eigen_dim_list:
        for neighbor in neighbor_list:
            train_x, train_y = train_set
            test_x, test_y = test_set

            if eigen_dim is None:
                continue
            else:
                train_x, test_x = eigenprojection(train_x, test_x, eigen_dim)

            train_x = train_x[:10000]
            test_x = test_x[:1000]
            train_y = train_y[:10000]
            test_y = test_y[:1000]

            print("Accuracy of kNN with k={} in {} eigen-dim: {:.4f}".format(neighbor, eigen_dim, kNN(train_x, train_y, test_x, test_y, neighbor)))


    # random forest
    print("Random Forest with different depth and estimators")
    train_x, train_y = train_set
    test_x, test_y = test_set

    train_x = train_x[:10000]
    test_x = test_x[:1000]
    train_y = train_y[:10000]
    test_y = test_y[:1000]

    forest_estimators_list = [10, 20, 30, 50, 80, 100, 200]
    forest_depth_list = [None, 10, 20, 30, 50, 100, 200, 500]

    for tree in forest_estimators_list:
        for depth in forest_depth_list:
            print("Accuracy of Random Forest with tree={} and depth={}: {:.4f}".format(tree, depth, random_forest(tree, depth, train_x, train_y, test_x, test_y)))