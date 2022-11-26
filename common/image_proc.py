import numpy as np

from common.neighbourhood import get_valid_neighbours


def expand_label(mat, label_mat, point, currentLabel):
    """
    Simple Breadth-First Search (BFS)
    :param mat:
    :param label_mat:
    :param point:
    :param currentLabel:
    :return:
    """
    expansionQueue = []

    expansionQueue.append(point)

    while expansionQueue:
        point = expansionQueue.pop(0)
        neighbours = get_valid_neighbours(point, np.shape(mat))

        for neighbour in neighbours:
            location = tuple(neighbour)

            if mat[location] != 0 and label_mat[location] == 0:
                label_mat[location] = currentLabel
                expansionQueue.append(location)

    return label_mat


def find_objects(mat):
    label_mat = np.zeros(mat.shape)

    label = 1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] != 0 and label_mat[i, j] == 0:
                label_mat[i, j] = label
                label_mat = expand_label(mat, label_mat, (i,j), label)
                label+=1

    return label_mat


def apply_mask(data, mask):
    test = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j] != 0:
                test[i, j] = data[i, j]

    return test