import numpy as np
import os


def find_file_directory(file_name):
    start = os.getcwd().split("approximate-inference")[0] + "approximate-inference"
    for root, dirs, files in os.walk(start):
        if file_name in files:
            return os.path.join(root, file_name)
    return None


def load_data():

    # load data
    temp = np.loadtxt(find_file_directory("co2.txt"))
    date_col = temp[:, 0] + ((temp[:, 1] - 1) / 12)
    X = np.hstack((date_col.reshape(-1, 1), np.ones(temp.shape[0]).reshape(-1, 1)))
    Y = temp[:, 2]

    return X, Y


def generate_data(N, D):
    # Define the basic shapes of the features
    m1 = [0, 0, 1, 0,
          0, 1, 1, 1,
          0, 0, 1, 0,
          0, 0, 0, 0]

    m2 = [0, 1, 0, 0,
          0, 1, 0, 0,
          0, 1, 0, 0,
          0, 1, 0, 0]

    m3 = [1, 1, 1, 1,
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0]

    m4 = [1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1]

    m5 = [0, 0, 0, 0,
          0, 0, 0, 0,
          1, 1, 0, 0,
          1, 1, 0, 0]

    m6 = [1, 1, 1, 1,
          1, 0, 0, 1,
          1, 0, 0, 1,
          1, 1, 1, 1]

    m7 = [0, 0, 0, 0,
          0, 1, 1, 0,
          0, 1, 1, 0,
          0, 0, 0, 0]

    m8 = [0, 0, 0, 1,
          0, 0, 0, 1,
          0, 0, 0, 1,
          0, 0, 0, 1]

    nfeat = 8  # number of features
    rr = 0.5 + np.random.rand(nfeat, 1) * 0.5  # weight of each feature between 0.5 and 1
    mut = np.array([rr[0] * m1, rr[1] * m2, rr[2] * m3, rr[3] * m4, rr[4] * m5,
                    rr[5] * m6, rr[6] * m7, rr[7] * m8])
    s = np.random.rand(N, nfeat) < 0.3  # each feature occurs with prob 0.3 independently

    # Generate Data - The Data is stored in Y
    Y = np.dot(s, mut) + np.random.randn(N, D) * 0.1  # some Gaussian noise is added
    return Y