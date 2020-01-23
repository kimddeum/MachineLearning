import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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
    train_x, train_y = train_set
    test_x, test_y = test_set

    return (train_x, train_y, test_x, test_y)


class dim_reduction:

    def __init__(self, data):
        self.train_x, self.train_y, self.test_x, self.test_y = data

    def LDA(self, dim):

        num_class = len(np.unique(self.train_y))
        length = self.train_x.shape[1]

        S_W = np.zeros((length, length))      # Variance within classes S_W
        S_B = np.zeros((length, length))      # Variance between class S_B

        global_mean = np.mean(self.train_x, axis=0)
        mean_vectors = []

        for c in range(num_class):
            class_data = self.train_x[self.train_y == c]
            class_mean = np.mean(class_data, axis=0)
            mean_vectors.append(class_mean)

            class_sub = np.zeros((length, length))

            for row in class_data:
                t = (row - class_mean).reshape(1, length)
                class_sub += np.dot(t.T, t)

            S_W += class_sub

            n = class_data.shape[0]
            class_mean = class_mean.reshape(1, length)
            diff = class_mean - global_mean

            S_B += n * np.dot(diff.T, diff)

        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.pinv(S_W), S_B))

        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        eigenvectors = eigenvectors.real.T[:dim].T

        projected_train = np.dot(self.train_x, eigenvectors)
        projected_test = np.dot(self.test_x, eigenvectors)


        if dim is 2:
            dim_reduction.scatter(self, projected_train, "LDA")

        return projected_train, projected_test

    def PCA(self, dim):

        cov = np.cov((self.train_x).T)

        _, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T[:dim].T

        projected_train = np.dot(self.train_x, eigenvectors)
        projected_test = np.dot(self.test_x, eigenvectors)

        if dim is 2:
            dim_reduction.scatter(self, projected_train, "PCA")

        return projected_train, projected_test

    def kNN(self, dim, mode):

        train_x, test_x = [], []

        if mode is 'PCA':
            train_x, test_x = dim_reduction.PCA(self, dim)
        elif mode is 'LDA':
            train_x, test_x = dim_reduction.LDA(self, dim)

        clf = KNeighborsClassifier().fit(train_x.real, self.train_y)

        train_accuracy = clf.score(train_x.real, self.train_y)
        test_accuracy = clf.score(test_x.real, self.test_y)

        dim_reduction.record(self, dim, mode, "kNN", train_accuracy, test_accuracy)

    def random_forest(self, dim, mode):

        train_x, test_x = [], []

        if mode is 'PCA':
            train_x, test_x = dim_reduction.PCA(self, dim)
        elif mode is 'LDA':
            train_x, test_x = dim_reduction.LDA(self, dim)

        clf = RandomForestClassifier().fit(train_x.real, self.train_y)

        train_accuracy = clf.score(train_x.real, self.train_y)
        test_accuracy = clf.score(test_x.real, self.test_y)

        dim_reduction.record(self, dim, mode, "Random Forest", train_accuracy, test_accuracy)

    def record(self, dim, mode, classification, train_accuracy, test_accuracy):

        f = open("./Result/record.txt", "a")
        f.write("[{}]\tdim: {}\tclassification: {}\n\ttraining accuracy: {:4f}, \ttesting accuracy: {:4f}\n".format(mode, dim, classification, train_accuracy, test_accuracy))
        f.close()

    def scatter(self, data, mode):

        fig, ax = plt.subplots()

        for c in range(10):
            class_data = np.array(data[self.train_y == c])
            ax.scatter(class_data[:, 0], class_data[:, 1], s=7, cmap='rainbow', label=c)

        ax.legend(fontsize=6, loc='upper left')
        fig.savefig("./Result/{}.png".format(mode))



if __name__ == '__main__':

    dim_list = [2, 3, 5, 9]
    mode = ["PCA", "LDA"]
    classification = ["kNN", "random_forest"]

    data = load_data('mnist.pkl.gz')

    reduction = dim_reduction(data)

    for dim in dim_list:
        for m in mode:
            for c in classification:

                if c is "kNN":
                    reduction.kNN(dim, m)

                elif c is "random_forest":
                    reduction.random_forest(dim, m)