import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from common.array_processing import split_consecutive, split_consecutive2


def plot_this_shit(plot, data, labels, peaks, subpeaks, boxes, contours):
    fig, ax = plt.subplots()

    for peak in peaks:
        ax.scatter(peak[0], peak[1], s=2, c='gray', marker='o')

    if subpeaks is not None:
        for subpeak in subpeaks:
            if subpeak not in peaks:
                ax.scatter(subpeak[0], subpeak[1], s=1, c='white', marker='o')


    im = ax.imshow(data, aspect='auto', cmap='jet', interpolation='none')

    # Get the current reference
    ax = plt.gca()

    for box in boxes:
        ax.add_patch(Rectangle((box[2], box[3]), box[0]-box[2], box[1]-box[3], linewidth=1, edgecolor='red', facecolor='none'))


    if contours is not None:
        for contour in contours:
            if contour is not None and contour.shape[0] != 0:
                sorted = contour[np.lexsort((contour[:, 0], contour[:, 1]))]
                split_contours = split_consecutive2(sorted, 0)
                test = []
                for cons in split_contours:
                    test.append(cons[::10])

                for contour_point in np.vstack(test):
                    ax.scatter(contour_point[0], contour_point[1], s=1, c='black', marker='o')

    plt.colorbar(im)
    ax.invert_yaxis()


    plt.savefig(f"fig2-{plot}.svg")
    plt.show()



data_folder = "../analysis_data/"
# data_file = "atoms-2"
# data_file = "eeg-t1"
data_file = "eeg-t1z"
# data_file = "rf-t12"

sbm_alg = '-sbm'
sbm_boxes_file          = data_file+sbm_alg+"-boxes"
sbm_contour_files       = data_file+sbm_alg+"-contour"
sbm_labels_file         = data_file+sbm_alg+"-image"
sbm_peaks_file          = data_file+sbm_alg+"-points-2"
sbm_subpeaks_file       = data_file+sbm_alg+"-points-1"

pf_alg = '-pf'
pf_boxes_file           = data_file+pf_alg+"-boxes"
pf_contour_files        = data_file+pf_alg+"-contour"
pf_labels_file          = data_file+pf_alg+"-image"
pf_peaks_file           = data_file+pf_alg+"-points-2"
pf_subpeaks_file        = data_file+pf_alg+"-points-1"

oe_alg = '-oe'
oe_labels_file          = data_file+oe_alg+"-image"
oe_boxes_file           = data_file+oe_alg+"-boxes"
oe_peaks_file           = data_file+oe_alg+"-points-2"

ext = ".csv"

data            = np.loadtxt(data_folder + data_file        + ext, delimiter=",", dtype=float, skiprows=5)






sbm_labels      = np.loadtxt(data_folder + sbm_labels_file      + ext, delimiter=",", dtype=int)
sbm_boxes       = np.loadtxt(data_folder + sbm_boxes_file       + ext, delimiter=",", dtype=int).T
sbm_peaks       = np.loadtxt(data_folder + sbm_peaks_file       + ext, delimiter=",", dtype=int).T
sbm_subpeaks    = np.loadtxt(data_folder + sbm_subpeaks_file    + ext, delimiter=",", dtype=int).T
sbm_contours = []
for file in os.listdir(data_folder):
    if file.startswith(sbm_contour_files):
        print(file)
        sbm_contours.append(np.loadtxt(data_folder + file, delimiter=",", dtype=int).T)



sbm_data = np.copy(data)
plot_this_shit(data_file+sbm_alg+"-cds", sbm_data, sbm_labels, sbm_peaks, sbm_subpeaks, sbm_boxes, sbm_contours)




pf_labels       = np.loadtxt(data_folder + pf_labels_file       + ext, delimiter=",", dtype=int)
pf_boxes        = np.loadtxt(data_folder + pf_boxes_file        + ext, delimiter=",", dtype=int).T
pf_peaks        = np.loadtxt(data_folder + pf_peaks_file        + ext, delimiter=",", dtype=int).T
pf_contours = []
for file in os.listdir(data_folder):
    if file.startswith(pf_contour_files):
        print(file)
        pf_contours.append(np.loadtxt(data_folder + file, delimiter=",", dtype=int).T)

try:
    pf_subpeaks = np.loadtxt(data_folder + pf_subpeaks_file + ext, delimiter=",", dtype=int).T

    pf_data = np.copy(data)
    plot_this_shit(data_file+pf_alg+"-cds", pf_data, pf_labels, pf_peaks, pf_subpeaks, pf_boxes, pf_contours)
except OSError:
    pf_data = np.copy(data)
    plot_this_shit(data_file+pf_alg+"-cds", pf_data, pf_labels, pf_peaks, None, pf_boxes, pf_contours)




oe_labels       = np.loadtxt(data_folder + oe_labels_file       + ext, delimiter=",", dtype=float)
oe_boxes        = np.loadtxt(data_folder + oe_boxes_file        + ext, delimiter=",", dtype=int).T
oe_peaks        = np.loadtxt(data_folder + oe_peaks_file        + ext, delimiter=",", dtype=int).T

oe_data = np.copy(data)
plot_this_shit(data_file+oe_alg+"-cds", oe_data, oe_labels, oe_peaks, None, oe_boxes, None)


