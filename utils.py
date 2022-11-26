import numpy as np

from common.kernels import gaussian_kernel
from preprocess.data_scaling import normalize_data_min_max


def save_csv(file, segm_data):
    f = open(file, "r")
    intro = f.readlines()[:5]
    f.close()

    f = open(file.replace(".csv", "")+"-segm.csv", "w")
    for line in intro:
        f.write(line)

    for i in range(len(segm_data)):
        for j in range(len(segm_data[0])):
            if j == len(segm_data[0]) - 1:
                f.write(str(segm_data[i, j]))
            else:
                f.write(str(segm_data[i, j]) + ", ")
        f.write("\n")
    f.close()


def toy_data():
    # ------------------A toy example------------------
    xx = np.linspace(-10, 10, 100)
    yy = np.linspace(-10, 10, 100)

    XX, YY = np.meshgrid(xx, yy)

    slab = np.zeros(XX.shape)

    # add 3 peaks
    slab += 5 * np.exp(-XX ** 2 / 1 ** 2 - YY ** 2 / 1 ** 2)
    slab += 8 * np.exp(-(XX - 3) ** 2 / 2 ** 2 - YY ** 2 / 2 ** 2)
    slab += 10 * np.exp(-(XX + 4) ** 2 / 2 ** 2 - YY ** 2 / 2 ** 2)

    return xx, XX, yy, YY, slab



def three_gaussian_data():
    data = np.zeros((100, 51))
    data[1:42, 5:-5] = data[1:42, 5:-5] + gaussian_kernel(40, 4)
    data[25:76] = data[25:76] + gaussian_kernel(50, 7)*5
    data[59:100, 5:-5] = data[59:100, 5:-5] + gaussian_kernel(40, 4)
    data = normalize_data_min_max(data) * 255

    return data