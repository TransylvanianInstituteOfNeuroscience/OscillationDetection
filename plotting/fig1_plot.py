import numpy as np
from matplotlib import pyplot as plt

from common.array_processing import split_consecutive
from algorithms.TFBM import run_TFBM


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


def plot1D(xx, line, labelsLine):
    plt.figure()
    plt.title('Result')

    test_split = split_consecutive(xx[labelsLine == 0], stepsize=1)
    print(test_split)

    for ts in test_split:
        plt.plot(ts, line[ts], 'b-')

    zord = -1
    plt.plot(xx[labelsLine == 1], line[labelsLine == 1], 'r-', zorder=zord)
    plt.plot(xx[labelsLine == 2], line[labelsLine == 2], 'g-', zorder=zord)
    plt.plot(xx[labelsLine == 3], line[labelsLine == 3], 'y-', zorder=zord)
    plt.plot(xx, np.full(len(xx), THR / 100 * np.amax(data)), 'k-', linestyle='dotted')

    for cc in cc_info:
        coords = cc['coordinates']
        # print(cc)
        # plt.annotate(f'{cc[1]}, {cc[0]}', xy=(cc[1], cc[0]), xycoords='data', xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
        plt.scatter(coords[0], line[coords[0]], s=2, c='black', marker='o', zorder=1)
        # if cc['parent'] == -1:
        #     plt.scatter(coords[0], coords[1], s=1, c='white', marker='o')

    plt.savefig("fig1.svg")
    plt.show()


xx, XX, yy, YY, slab = toy_data()

xx = np.arange(len(xx))

line = slab[slab.shape[0] // 2]
print(line)
plt.figure()
plt.title('1D presentation data')
plt.plot(xx, line, 'b-')
plt.show()

THR = 5
G_PULL = 1
EXP_FACTOR = 3

data = line[:, None]

labelsMatrix, cc_info = run_TFBM(data,
                                 threshold=THR / 100 * np.amax(data),
                                 gravitational_pull=G_PULL,
                                 expansion_factor=EXP_FACTOR,
                                 scale=np.array([1 ,1]),
                                 disambig="yes", merging="yes")
print(np.unique(labelsMatrix))

labelsLine = np.squeeze(labelsMatrix)

plot1D(xx, line, labelsLine)