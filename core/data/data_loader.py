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
