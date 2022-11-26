import os

import numpy as np
from matplotlib import pyplot as plt

from common.image_proc import apply_mask
from preprocess.data_scaling import normalize_data_min_max
from TFBM import cluster_center_bfs

def run_and_plot(name, data, threshold, gravitational_pull, scale, disambig, merging):

    fig= plt.figure(figsize=(24, 6))
    ax = fig.add_subplot(1, 3, 1)

    ax.set_title("Initial Data")
    im = ax.imshow(data, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    labelsMatrix, cc_info = cluster_center_bfs(data, threshold=threshold, gravitational_pull=gravitational_pull, scale=scale,
                                               disambig=disambig, merging=merging)
    print(f"{len(np.unique(labelsMatrix))} blobs found, containing {len(cc_info)} local maxima")


    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Labels of Data")
    im = ax.imshow(labelsMatrix, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)
    for cc in cc_info:
        coords = cc['coordinates']

        plt.scatter(coords[1], coords[0], s=1, c='black', marker='o')
        if cc['parent'] == -1:
            plt.scatter(coords[1], coords[0], s=1, c='white', marker='o')

    masked = apply_mask(data, labelsMatrix)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Zorro")
    im = ax.imshow(masked, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    for cc in cc_info:
        contour_points = cc['contour']
        for contour_point in contour_points:
            plt.scatter(contour_point[1], contour_point[0], s=1, c='black', marker='o')

    result_name = f"{name}_scale({scale[0]:.2f}, {scale[1]:.2f})thr{threshold:.2f}_gpull{gravitational_pull}_{'redistrib' if disambig == 'yes' else 'noredistrib'}_{'merge' if merging == 'yes' else 'nomerged'}"
    plt.savefig(f"figures/{result_name}.png", format='png')
    plt.show()





# _, _, _, _, data = toy_data()
# run_and_plot(data)

data_folder = "./data/"

for file in os.listdir(data_folder):
    if not file.endswith("-segm.csv") and file.endswith(".csv") and file.startswith("atoms-2.csv"):
        print(file)
        data = np.loadtxt(data_folder + file, delimiter=",", dtype=float, skiprows=5)
        print(data.shape)
        signal_length = data.shape[1]

        downsampling_factor = 10
        data = normalize_data_min_max(data) * 100
        data = data[:, ::downsampling_factor]

        scale = np.array(data.shape) / max(data.shape)
        # scale = 100 * normalize_data_min_max(data.shape)
        scale = 100 / np.array(data.shape)
        print(scale)

        # scale = np.array([1.0, 1.0])

        THR = 10
        G_PULL = 1

        run_and_plot(file.replace(".csv", ""), data, threshold=THR / 100 * np.amax(data), gravitational_pull=G_PULL, scale=scale, disambig="yes", merging="yes")

