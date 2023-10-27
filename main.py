import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from algorithms.OEvents import OEvents
from algorithms.TFBM import TFBM
from algorithms.TFPF import TFPF
from algorithms.structs import Spectrum2D
from common.image_proc import apply_mask


def plot_data_result_mask(method_name, data, labelsMatrix, center_coords):
    fig= plt.figure(figsize=(24, 6))
    fig.suptitle(method_name, fontsize=16)
    ax = fig.add_subplot(1, 3, 1)

    ax.set_title("Initial Data")
    im = ax.imshow(data, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Mask")
    im = ax.imshow(labelsMatrix, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    for center_coord in center_coords:
        plt.scatter(center_coord[1], center_coord[0], s=1, c='black', marker='o')
        # if cc['parent'] == -1:
        #     plt.scatter(coords[1], coords[0], s=1, c='white', marker='o')

    masked = apply_mask(data, labelsMatrix)

    # test = np.zeros_like(data)
    # for cc in cc_info:
    #     contour_points = cc['contour']
    #     for contour_point in contour_points:
    #             test[contour_point[0], contour_point[1]] = 1

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Segmentation")
    im = ax.imshow(masked, aspect='auto', cmap='jet', interpolation='none')
    plt.colorbar(im)
    # im = ax.imshow(test, aspect='auto', cmap='bone_r', interpolation='none', alpha=test)
    ax.invert_yaxis()


    plt.show()

def TFBM_and_plot(data):
    tfbm = TFBM(data.T, threshold="auto", merge=True, merge_factor=15)
    tfbm.fit(verbose=True, timer=True)

    center_coords = [(pi.center_coords[1], pi.center_coords[0]) for pi in tfbm.packet_infos]

    plot_data_result_mask("TFBM", data, tfbm.merged_labels_data.T, center_coords)



def TFPF_and_plot(data):
    sliceLevel=10

    peakfinder = TFPF()
    peakfinder.evaluate(data)
    peakfinder.slice_n_dice(sliceLevel=sliceLevel)

    labelsMatrix = peakfinder.labels

    center_coords = [peak.coordinates for peak in peakfinder.peaks]

    plot_data_result_mask("TFPF", data, labelsMatrix, center_coords)


def OEvents_and_plot(spectrumData):
    oe = OEvents()
    oe.evaluateMedianThr(spectrumData)

    events = oe.events

    data = spectrumData.powerValues

    fig= plt.figure(figsize=(24, 6))
    fig.suptitle("OEvents", fontsize=16)
    ax = fig.add_subplot(1, 2, 1)

    ax.set_title("Initial Data")
    im = ax.imshow(data, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Events Found")
    im = ax.imshow(data, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    for event in events:
        box = event.boundingBox
        ax.add_patch(Rectangle((box.R, box.B), box.L-box.R, box.T-box.B, linewidth=1, edgecolor='red', facecolor='none'))

    plt.show()



def load_atoms_synthetic_data():
    data_folder = "./DATA/toy/"
    file = "atoms-2.csv"

    f = open(data_folder+file, "r")
    intro = f.readlines()[:5]
    f.close()

    timeValues = []
    for str_time in intro[1].split(","):
        timeValues.append(float(str_time))

    frequencyValues = []
    for str_time in intro[3].split(","):
        frequencyValues.append(float(str_time))

    data = np.loadtxt(data_folder + file, delimiter=",", dtype=float, skiprows=5)

    spectrumData = Spectrum2D(timeValues=np.array(timeValues), frequencyValues=frequencyValues, powerValues=data)

    return data, spectrumData


def load_downsampled_atoms_synthetic_data():
    data_folder = "./DATA/toy/"
    file = "atoms-2.csv"

    f = open(data_folder+file, "r")
    intro = f.readlines()[:5]
    f.close()

    timeValues = []
    for str_time in intro[1].split(","):
        timeValues.append(float(str_time))

    frequencyValues = []
    for str_time in intro[3].split(","):
        frequencyValues.append(float(str_time))

    data = np.loadtxt(data_folder + file, delimiter=",", dtype=float, skiprows=5)

    downsampling_factor = 10
    data = data[:, ::downsampling_factor]

    return data



if __name__ == "__main__":
    data, spectrumData = load_atoms_synthetic_data()
    OEvents_and_plot(spectrumData)
    TFBM_and_plot(data)
    TFPF_and_plot(data)

