import random
from collections import deque

import numpy as np
import warnings

from scipy.ndimage import maximum_filter
from sklearn.preprocessing import LabelEncoder

from common.distance import euclidean_point_distance, euclidean_point_distance_scale, euclidean_points_distance
from common.neighbourhood import get_valid_neighbours8

warnings.simplefilter(action='ignore', category=FutureWarning)


def run_TFBM(data, threshold, gravitational_pull, expansion_factor, scale, disambig=False, merging=False):
    """
    Main function for running the algorithm
    :param data: matrix - spectrogram of the power values
    :param threshold: float - algorithm parameter
    :param gravitational_pull: float - algorithm parameter
    :param expansion_factor: float - algorithm parameter
    :param scale: vector of 2 - algorithm parameter
    :param disambig: boolean - algorithm parameter
    :param merging: boolean - algorithm parameter
    :return:
    """
    clusterCenters = find_cluster_centers_no_neighbours(data, threshold=threshold)

    pairs = []
    for cc in clusterCenters:
        pairs.append([cc, data[tuple(cc)]])

    pairs = np.array(pairs)

    cc_info = []
    sorted_pairs = pairs[pairs[:, 1].argsort()][::-1]
    for pair in sorted_pairs:
        test = {}
        test['coordinates'] = tuple(pair[0])
        test['peak'] = pair[1]
        cc_info.append(test)


    labelsMatrix = np.zeros_like(data, dtype=int)
    for index in range(len(cc_info)):
        point = tuple(cc_info[index]['coordinates'])
        if labelsMatrix[point] != 0:
            continue  # cluster was already discovered
        labelsMatrix = expand_cluster_center(data, point, labelsMatrix, index+1, cc_info, expansion_factor, scale=scale, disambig=disambig)


    for id, cc in enumerate(cc_info):
        contour = []
        testMatrix = np.zeros_like(labelsMatrix)
        for point in cc['points']:
            testMatrix[point] = 1
        for point in cc['points']:
            neighbours = get_valid_neighbours8(point, np.shape(data))
            for neighbour in neighbours:
                if testMatrix[tuple(neighbour)] == 0:
                    contour.append(point)
                    break
        cc_info[id]['contour'] = contour

    for id, cc in enumerate(cc_info):
        points3D = []
        points2D = cc['contour']
        for point in points2D:
            points3D.append([point[0], point[1], data[point]])

        cc_info[id]['prominence'] = cc_info[id]['peak'] - np.amax(np.array(points3D)[:, 2])

    if merging == True:
        cc_info, labelsMatrix = merge_labels(cc_info, data, labelsMatrix, scale, gravitational_pull)


    le = LabelEncoder()
    labelsMatrix = le.fit_transform(labelsMatrix.reshape(-1, 1))
    labelsMatrix = labelsMatrix.reshape(data.shape)


    for i in range(len(cc_info)):
        if cc_info[i]['parent'] == -1:
            print("Center Coords:", cc_info[i]['coordinates'])
            print("Peak Value:", cc_info[i]['peak'])
            print(f"Prominence : {cc_info[i]['prominence'] :.2f}")
            print("Parent Cluster:", cc_info[i]['parent'])
            print("Start Label:", cc_info[i]['start_label'])
            print("Finish Label:", cc_info[i]['finish_label'])
            print()
            print()


    label_list = list(range(1, len(np.unique(labelsMatrix))))
    random.shuffle(label_list)

    newLabelsMatrix = np.zeros_like(labelsMatrix)
    for id, label in enumerate(np.unique(labelsMatrix)[1:]):
        newLabelsMatrix[labelsMatrix == label] = label_list[id]

    return newLabelsMatrix, cc_info



def merge_labels(cc_info, array, labels, scale, G_PULL = 1.5):
    """
    Post-Process for the merging of  labels
    :param cc_info: list of dicts - internal structure for holding data
    :param array: matrix - an array of the power values, representing the spectrogram
    :param labels: matrix - the label given by the algorithm for each power value
    :param scale: vector of 2 - algorithm parameter
    :param G_PULL: float - algorithm parameter
    :return:
    """
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





def find_cluster_centers_no_neighbours(array, threshold=5):
    """
    Search through the spectrogram matrix to find the cluster centers
    :param array:       matrix - an array of the power values, representing the spectrogram
    :param threshold:   integer - oscillation packet threshold, minimum value needed for a (x,y) point to be considered a possible oscillation packet center

    :returns clusterCenters: vector - a vector of the coordinates of the chunks that are cluster centers
    """

    clusterCenters = np.argwhere((maximum_filter(array, size=3) == array) & (array > threshold))

    return clusterCenters




def get_dropoff(array, location):
    """
    Calculate the dropoff of a certain location
    :param array:       matrix - an array of the power values, representing the spectrogram
    :param location:    tuple - indicates the (x,y) of the current point for which the dropoff will be calculated

    :return: dropoff:   float - a float value that represents the dropoff of a certain location
    """
    neighbours = get_valid_neighbours8(location, array.shape)
    dropoff = np.sum((array[location] - array[neighbours[:, 0], neighbours[:, 1]]) ** 2) / len(neighbours)
    if dropoff > 0:
        return np.sqrt(dropoff / len(neighbours)) / array[location]
    return 0



def expand_cluster_center(array, start, labels, currentLabel, cc_info, expansion_factor, scale, disambig=False):  # TODO
    """
    Expansion
    :param array: matrix - an array of the power values, representing the spectrogram
    :param start: tuple - the coordinates of the (x,y) power value where the expansion starts (current cluster center)
    :param labels: matrix - the labels array
    :param currentLabel: integer - the label of the current oscillation packet
    :param cc_info: vector - vector of dictionary that represents the structure that holds information of each detected oscillation packet
    :param expansion_factor: float - algorithm parameter
    :param scale: vector of 2 - algorithm parameter
    :param disambig: boolean - whether the algorithm will disambiguate certain points that had conflicts

    :returns labels: matrix - updated matrix of labels after expansion and conflict solve
    """
    visited = np.zeros_like(array, dtype=bool)
    expansionQueue = deque()
    if labels[start] == 0:
        expansionQueue.append(start)
        labels[start] = currentLabel

    visited[start] = True

    cc_index = 0
    for id, cc in enumerate(cc_info):
        if tuple(cc['coordinates']) == start:
            cc_index = id
            break

    dropoff = get_dropoff(array, start)

    conflicts = {}
    points = []
    cc_info[cc_index]['parent'] = -1
    cc_info[cc_index]['start_label'] = currentLabel
    cc_info[cc_index]['finish_label'] = currentLabel

    while expansionQueue:
        point = expansionQueue.popleft()
        points.append(tuple(point))
        neighbours = get_valid_neighbours8(point, array.shape)

        distances = euclidean_points_distance(start, neighbours, scale)

        for neigh_id, neighbour in enumerate(neighbours):
            location = tuple(neighbour)
            number = dropoff * distances[neigh_id]

            if (not visited[location]) and (number * expansion_factor < array[location] <= array[point]):
                visited[location] = True
                expansionQueue.append(location)

                if labels[location] == 0:
                    labels[location] = currentLabel
                else:
                    oldLabel = labels[location]

                    key = f"{cc_info[oldLabel-1]['coordinates']}"
                    if key in conflicts.keys():
                        conflicts[key].append(location)
                    else:
                        conflicts[key] = []
                        conflicts[key].append(location)

                    if disambig==True:
                        disRez = disambiguate(array,
                                              location,
                                              cc_info[currentLabel - 1]['coordinates'],
                                              cc_info[oldLabel - 1]['coordinates'])
                        if disRez == 1:
                            labels[location] = currentLabel
                        elif disRez == 2:
                            labels[location] = oldLabel

    cc_info[cc_index]['zconflicts'] = conflicts
    cc_info[cc_index]['points'] = points

    return labels


def disambiguate(array, questionPoint, clusterCenter1, clusterCenter2):
    """
    Disambiguation of a point from the spectrogram based on the parameters
    :param array: matrix - an array of the power values, representing the spectrogram
    :param questionPoint: tuple - the coordinates (x, y) in the spectrogram toward which the expansion is going
    :param clusterCenter1: tuple - the coordinates (x, y) of the power value of the first oscillation packet
    :param clusterCenter2: tuple - the coordinates (x, y) of the power value of the second oscillation packet

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


