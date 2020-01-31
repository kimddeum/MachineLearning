import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans


class RBFN:

    def __init__(self, num_basis, mode):
        self.num_basis = num_basis
        self.mode = mode
        self.weight = np.zeros(num_basis)
        self.means = np.zeros((num_basis, 1))
        self.stds = np.zeros(num_basis)

    def calculate_rbf(self, x, u, s):
        return np.exp(-1 * (np.linalg.norm(x-u)**2 / s))

    def fit(self, data, label):
        self.means = KMeans(self.num_basis, random_state=0).fit(data.reshape(-1, 1)).cluster_centers_

        dMax = np.max([np.abs(c1 - c2) for c1 in self.means for c2 in self.means])
        self.stds = np.repeat(dMax / np.sqrt(2*self.num_basis), self.num_basis)

        b = np.zeros((len(self.means), data.shape[0]))
        for i in range(len(self.means)):
            for j in np.ndindex(data.shape[0]):
                b[i][j] = RBFN.calculate_rbf(self, data[j], self.means[i], self.stds[i])

        self.weight = np.dot(np.linalg.pinv(b).T, label)

        return self

    def predict(self, data, real):

        b = np.zeros((len(self.means), data.shape[0]))

        for i in range(len(self.means)):
            for j in np.ndindex(data.shape[0]):
                b[i][j] = RBFN.calculate_rbf(self, data[j], self.means[i], self.stds[i])

        predicted = np.dot(b.T, self.weight)

        if self.mode is 'cis':

            sigmoid = 1/(1+np.exp(- predicted))
            sigmoid[sigmoid > 0.5] = 1
            sigmoid[sigmoid <= 0.5] = 0

            accuracy = np.equal(sigmoid, real).mean()

            return predicted, accuracy

        elif self.mode is 'fa':

            mse = ((real - predicted)**2).mean()

            return predicted, mse


def draw_loss(t, basis, lst, fa, cis):
    plt.plot(basis, lst)

    if t in fa:
        plt.title('Function Approximation - MSE')
        plt.xlabel('number of basis functions')
        plt.ylabel('MSE')

        plt.savefig('./Results/{}_MSE.png'.format(t.split(".")[0]))

    elif t in cis:
        plt.title('Circle in the square - Accuracy')
        plt.xlabel('number of basis functions')
        plt.ylabel('Accuracy')

        plt.savefig('./Results/{}_Accuracy.png'.format(t.split(".")[0]))

    plt.close()


def run(t, basis_num, fa, cis):
    lst = []

    for n in basis_num:
        train_x, train_y = read_file(get_path(t))
        test_x, test_y = read_file(get_path(test))

        rbfn = RBFN(n, m).fit(train_x, train_y)
        predicted, loss = rbfn.predict(test_x, test_y)
        lst.append(loss)

    draw_loss(t, basis_num, lst, fa, cis)


def get_path(text):
    dir = os.getcwd()
    return os.path.join(os.path.join(dir, "RBFN_train_test_files"), str(text))


def read_file(name):
    train, label = [], []

    with open(name, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.split("\t")
            if len(data) == 3:                  # circle in the square
                train.append([float(data[0]), float(data[1])]), label.append(int(data[2]))
            else:                               # function approximation
                train.append(float(data[0])), label.append(float(data[1]))

    f.close()

    return np.asarray(train), np.asarray(label)


if __name__ == '__main__':

    mode = ['cis']
    #mode = ['fa', 'cis']
    cis = ['cis_train1.txt', 'cis_train2.txt', 'cis_test.txt']
    fa = ['fa_train1.txt', 'fa_train2.txt', 'fa_test.txt']

    basis_num = list(range(1, 21))

    for m in mode:
        if m is 'cis':
            train, test = cis[:2], cis[2]
        elif m is 'fa':
            train, test = fa[:2], fa[2]

        for t in train:
            run(t, basis_num, fa, cis)



