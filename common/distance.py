import numpy as np

def euclidean_point_distance(pointA, pointB):
    """
    Calculates the euclidean distance between 2 points (L2 norm/distance) for n-dimensional points
    :param pointA: vector - vector containing all the dimensions of a point A
    :param pointB: vector - vector containing all the dimensions of a point B

    :returns dist: float - the distance between the 2 points
    """
    difference = np.subtract(pointA, pointB)
    squared = np.square(difference)
    dist = np.sqrt(np.sum(squared))
    return dist


def euclidean_point_distance_scale(pointA, pointB, scale):
    """
    Calculates the euclidean distance between 2 points (L2 norm/distance) for n-dimensional points
    :param pointA: vector - vector containing all the dimensions of a point A
    :param pointB: vector - vector containing all the dimensions of a point B

    :returns dist: float - the distance between the 2 points
    """
    # scaledA = (pointA[0]*scale[0], pointA[1]*scale[1])
    # scaledB = (pointB[0]*scale[0], pointB[1]*scale[1])
    return euclidean_point_distance(pointA*scale, pointB*scale)


def euclidean_point_mat_distance(point, mat):
    """
    Calculates the euclidean distance between 2 points (L2 norm/distance) for n-dimensional points
    :param pointA: vector - vector containing all the dimensions of a point A
    :param pointB: vector - vector containing all the dimensions of a point B

    :returns dist: float - the distance between the 2 points
    """
    difference = np.subtract(mat, point)
    squared = np.square(difference)
    dist = np.sqrt(np.sum(squared, axis=-1))
    return dist