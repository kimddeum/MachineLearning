import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
# import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set

    # Calculate and draw mean of train_x
    mean = np.mean(train_x, axis=0)  # mean = train_x.mean(0)
    reshaped_mean = mean.reshape((28, 28)) * 255.9
    plt.imsave('mean.jpg', reshaped_mean, cmap='gray')
    plt.close()

    # Calculate and draw variance of train_x
    var = np.var(train_x, axis=0)  # var = train_x.var(0)
    reshaped_var = var.reshape((28, 28)) * 255.9
    plt.imsave('var.jpg', reshaped_var, cmap='gray')
    plt.close()
    
    # Calculate eigenvalues and eigenvectors using covariance matrix
    cov = np.cov(train_x.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvectors = eigenvectors.T

    # Draw first 10 eigenvectors
    for i in range(10):
        reshaped_eig_v = eigenvectors[i].reshape((28, 28)) * 255.9
        plt.imsave('eigen_vector'+str(i)+'.jpg', reshaped_eig_v.real, cmap='gray')


    # Draw plot of first 100 eigenvalues
    plt.plot(range(100), eigenvalues[:100])
    plt.savefig('eig_val.jpg')
