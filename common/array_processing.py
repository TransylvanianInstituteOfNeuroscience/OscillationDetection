import numpy as np

def split_consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def split_consecutive2(data, col, stepsize=1):
    return np.split(data, np.where(np.logical_and(np.diff(data[:, col]) != stepsize, np.diff(data[:, col]) != 0))[0]+1)