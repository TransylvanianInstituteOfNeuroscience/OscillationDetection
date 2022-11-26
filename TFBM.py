import math
import sys

import random

import numpy as np
import warnings

from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from skspatial.objects import Plane

from common.distance import euclidean_point_distance, euclidean_point_distance_scale
from common.geometry import plane_from_points, point_to_plane_distance, line_from_two_points, point_to_line_distance, \
    plane_from_three_points, perpendicular_plane_from_plane_and_two_points
from common.maxima import check_maxima_no_neighbour_maxim, check_maxima
from common.neighbourhood import get_valid_neighbours

sys.setrecursionlimit(1000000)

warnings.simplefilter(action='ignore', category=FutureWarning)


def cluster_center_bfs(data, threshold, gravitational_pull, scale, disambig="no", merging="no"):

    clusterCenters = find_cluster_centers_no_neighbours(data, threshold=threshold)

    pairs = []
    for cc in clusterCenters:
        pairs.append([cc, data[tuple(cc)]])

    pairs = np.array(pairs)

    # clusterCenters = sorted_pairs[:, 0]
    cc_info = []
    sorted_pairs = pairs[pairs[:, 1].argsort()][::-1]
    for pair in sorted_pairs:
        test = {}
        test['coordinates'] = pair[0]
        test['peak'] = pair[1]
        cc_info.append(test)





    labelsMatrix = np.zeros_like(data, dtype=int)
    for index in range(len(cc_info)):
        point = cc_info[index]['coordinates']
        if labelsMatrix[point] != 0:
            continue  # cluster was already discovered
        labelsMatrix = expand_cluster_center(data, point, labelsMatrix, index+1, cc_info,
                                             scale=scale, disambig=disambig)

    if merging == 'yes':
        cc_info, labelsMatrix = merge_labels(cc_info, data, labelsMatrix, scale, gravitational_pull)


    le = LabelEncoder()
    labelsMatrix = le.fit_transform(labelsMatrix.reshape(-1, 1))
    labelsMatrix = labelsMatrix.reshape(data.shape)






    for id, cc in enumerate(cc_info):
        contour = []
        testMatrix = np.zeros_like(labelsMatrix)
        for point in cc['points']:
            testMatrix[point] = 1
        for point in cc['points']:
            neighbours = get_valid_neighbours(point, np.shape(data))
            for neighbour in neighbours:
                if testMatrix[tuple(neighbour)] == 0:
                    contour.append(point)
                    break
        cc_info[id]['contour'] = contour


    for id, cc in enumerate(cc_info):
        center = cc['coordinates']
        center_point = [center[0], center[1], data[center]]
        points3D = []
        points2D = cc['contour']
        for point in points2D:
            points3D.append([point[0], point[1], data[point]])

        cc_info[id]['prominence'] = cc_info[id]['peak'] - np.amax(np.array(points3D)[:, 2])


    label_list = list(range(1, len(np.unique(labelsMatrix))))
    random.shuffle(label_list)

    newLabelsMatrix = np.zeros_like(labelsMatrix)
    for id, label in enumerate(np.unique(labelsMatrix)[1:]):
        newLabelsMatrix[labelsMatrix == label] = label_list[id]


    return newLabelsMatrix, cc_info



def merge_labels(cc_info, array, labels, scale, G_PULL = 1.5):
    for i in range(len(cc_info)):
        current_center = cc_info[i]['coordinates']
        current_label = cc_info[i]['finish_label']

        for conflict_center_str in list(cc_info[i]['zconflicts'].keys()):
            conflict_center = eval(conflict_center_str)
            conflict_label = labels[conflict_center]

            c1_pull_sum = 0
            c2_pull_sum = 0
            conflicts_coords = cc_info[i]['zconflicts'][conflict_center_str]
            for conflict_coord in conflicts_coords:
                distanceC1 = euclidean_point_distance_scale(current_center, conflict_coord, scale)
                distanceC2 = euclidean_point_distance_scale(conflict_center, conflict_coord, scale)
                c1_pull = array[current_center] * get_dropoff(array, current_center) * distanceC1
                c2_pull = array[conflict_center] * get_dropoff(array, conflict_center) * distanceC2

                c1_pull_sum+=c1_pull
                c2_pull_sum+=c2_pull

            if conflict_label == current_label:
                continue


            if c1_pull_sum > c2_pull_sum*G_PULL:
                labels[labels == conflict_label] = current_label

                cc_info[conflict_label - 1]['parent'] = cc_info[i]['coordinates']
                cc_info[conflict_label - 1]['finish_label'] = current_label


            elif c1_pull_sum*G_PULL < c2_pull_sum:
                labels[labels == current_label] = conflict_label

                cc_info[i]['parent'] = cc_info[conflict_label - 1]['coordinates']
                cc_info[i]['finish_label'] = conflict_label

    return cc_info, labels



def find_supreme_parent(cc_info, i):
    while (cc_info[i]['parent'] != -1):
        for id, cc in enumerate(cc_info):
            if cc_info[i]['parent'] == cc['coordinates']:
                i = id
                break
    return i



def find_cluster_centers(array, threshold=5):
    """
    Search through the matrix of chunks to find the cluster centers
    :param array: matrix - an array of the values in each chunk
    :param threshold: integer - cluster center threshold, minimum amount needed for a chunk to be considered a possible cluster center

    :returns clusterCenters: vector - a vector of the coordinates of the chunks that are cluster centers
    """
    clusterCenters = []

    for index, value in np.ndenumerate(array):
        if value >= threshold and check_maxima(array, index):  # TODO exclude neighbour centers
            clusterCenters.append(index)

    return clusterCenters




def find_cluster_centers_no_neighbours(array, threshold=5):
    """
    Search through the matrix of chunks to find the cluster centers
    :param array: matrix - an array of the values in each chunk
    :param threshold: integer - cluster center threshold, minimum amount needed for a chunk to be considered a possible cluster center

    :returns clusterCenters: vector - a vector of the coordinates of the chunks that are cluster centers
    """
    clusterCenters = []

    for index, value in np.ndenumerate(array):
        if value >= threshold and check_maxima_no_neighbour_maxim(array, index, clusterCenters):
            clusterCenters.append(index)

    return clusterCenters


def get_dropoff(ndArray, location):
    neighbours = get_valid_neighbours(location, np.shape(ndArray))
    dropoff = 0
    for neighbour in neighbours:
        neighbourLocation = tuple(neighbour)
        dropoff += ((ndArray[location] - ndArray[neighbourLocation]) ** 2)
    if dropoff > 0:
        return math.sqrt(dropoff / len(neighbours)) / ndArray[location]
    return 0


def get_falloff(ndArray, location):
    neighbours = get_valid_neighbours(location, np.shape(ndArray))
    dropoff = 0
    for neighbour in neighbours:
        neighbourLocation = tuple(neighbour)
        dropoff += ((ndArray[location] - ndArray[neighbourLocation]) ** 2)
    if dropoff > 0:
        return dropoff / len(neighbours)
    return 0


def get_strength(ndArray, clusterCenter, questionPoint):
    dist = euclidean_point_distance(clusterCenter, questionPoint)

    strength = ndArray[questionPoint] / dist / ndArray[clusterCenter]

    return strength


def expand_cluster_center(array, start, labels, currentLabel, cc_info, scale, disambig="no"):  # TODO
    """
    Expansion
    :param array: matrix - an array of the values in each chunk
    :param start: tuple - the coordinates of the chunk where the expansion starts (current cluster center)
    :param labels: matrix - the labels array
    :param currentLabel: integer - the label of the current cluster center
    :param clusterCenters: vector - vector of all the cluster centers, each containing n-dimensions
    :param version: integer - the version of SBM (0-original version, 1=license, 2=modified with less noise)

    :returns labels: matrix - updated matrix of labels after expansion and conflict solve
    """
    visited = np.zeros_like(array, dtype=bool)
    expansionQueue = []
    if labels[start] == 0:
        expansionQueue.append(start)
        labels[start] = currentLabel

    visited[start] = True

    cc_index = 0
    for id, cc in enumerate(cc_info):
        if cc['coordinates'] == start:
            cc_index = id
            break


    conflicts = {}
    points = []
    cc_info[cc_index]['parent'] = -1
    cc_info[cc_index]['start_label'] = currentLabel
    cc_info[cc_index]['finish_label'] = currentLabel
    while expansionQueue:
        point = expansionQueue.pop(0)
        points.append(tuple(point))
        neighbours = get_valid_neighbours(point, np.shape(array))

        for neighbour in neighbours:
            location = tuple(neighbour)

            number = get_dropoff(array, location) * math.sqrt(euclidean_point_distance_scale(start, location, scale))

            if (not visited[location]) and (number  < array[location] <= array[point]):
                visited[location] = True
                if labels[location] == currentLabel:
                    expansionQueue.append(location)
                elif labels[location] == 0:
                    expansionQueue.append(location)
                    labels[location] = currentLabel
                else:
                    oldLabel = labels[location]

                    key = f"{cc_info[oldLabel-1]['coordinates']}"
                    if key in conflicts.keys():
                        conflicts[key].append(location)
                    else:
                        conflicts[key] = []
                        conflicts[key].append(location)

                    if disambig=="yes":
                        disRez = disambiguate(array,
                                              location,
                                              point,
                                              cc_info[currentLabel - 1]['coordinates'],
                                              cc_info[oldLabel - 1]['coordinates'],
                                              scale)
                        # # print(currentLabel, oldLabel, disRez)
                        if disRez == 1:
                            labels[location] = currentLabel
                            expansionQueue.append(location)
                        elif disRez == 2:
                            labels[location] = oldLabel
                            expansionQueue.append(location)

    cc_info[cc_index]['zconflicts'] = conflicts
    cc_info[cc_index]['points'] = points

    return labels


def disambiguate(array, questionPoint, expansionPoint, clusterCenter1, clusterCenter2, scale):
    """
    Disambiguation of the cluster of a chunk based on the parameters
    :param array: matrix - an array of the values in each chunk
    :param questionPoint: tuple - the coordinates of the chunk toward which the expansion is going
    :param expansionPoint: tuple - the coordinates of the chunk from which the expansion is going
    :param clusterCenter1: tuple - the coordinates of the chunk of the first cluster center
    :param clusterCenter2: tuple - the coordinates of the chunk of the second cluster center
    :param version: integer - the version of SBM (0-original version, 1=license, 2=modified with less noise)

    :returns : integer - representing the approach to disambiguation
    """

    distanceToC1 = euclidean_point_distance(questionPoint, clusterCenter1)
    distanceToC2 = euclidean_point_distance(questionPoint, clusterCenter2)

    c1_pull = array[clusterCenter1] / array[questionPoint] - get_dropoff(array, clusterCenter1) * distanceToC1
    c2_pull = array[clusterCenter2] / array[questionPoint] - get_dropoff(array, clusterCenter2) * distanceToC2

    if c1_pull > c2_pull:
        return 1
    else:
        return 2


