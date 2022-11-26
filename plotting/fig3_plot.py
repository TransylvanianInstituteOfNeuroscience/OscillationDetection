import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd


TESTS = 4

METHODS = ['TFBM', 'TFPF', 'COE',  'ROE']
snr_values = [0.5, 1, 2, 4]
box_colors = ['darkkhaki', 'royalblue', 'pink', 'lightgreen']
weights = ['bold', 'semibold', 'light', 'heavy']


df = pd.read_csv('../analysis_data/analysis2-data-test.csv')
print(df)



def git(df, column, snr):
    return df.query(f"SNR == {snr}")[column].to_numpy()







def plot_box(data):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    c = 'k'
    black_dict = {  # 'patch_artist': True,
        # 'boxprops': dict(color=c, facecolor=c),
        # 'capprops': dict(color=c),
        # 'flierprops': dict(color=c, markeredgecolor=c),
        'medianprops': dict(color=c),
        # 'whiskerprops': dict(color=c)
    }

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5, showfliers=False, **black_dict)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Test this shit',
        xlabel='SNR',
        ylabel='Value',
    )

    # Now fill the boxes with desired colors
    num_boxes = len(data)
    for i in range(num_boxes):

        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])

        # Alternate among colors
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % len(METHODS)]))

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(np.repeat(snr_values, len(METHODS)), rotation=25, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    # pos = np.arange(num_boxes) + 1
    for id, (method, y) in enumerate(zip(METHODS, np.arange(0.01, 0.03 * len(METHODS), 0.03).tolist())):
        fig.text(0.90, y, METHODS[id],
                 backgroundcolor=box_colors[id],
                 color='black', weight='roman', size='x-small')

    plt.show()



box_errors = ['SBM-MatchErrBox', 'PF-MatchErrBox', 'COE-MatchErrBox', 'ROE-MatchErrBox']
data_box_error = []
for snr in snr_values:
    for column in box_errors:
        data_box_error.append(git(df, column, snr))
plot_box(data_box_error)


time_box_errors = ['SBM-TimeErrMask', 'PF-TimeErrMask', 'COE-TimeErrMask', 'ROE-TimeErrMask']
data_time_box_error = []
for snr in snr_values:
    for column in time_box_errors:
        data_time_box_error.append(git(df, column, snr))
plot_box(data_time_box_error)


freq_box_errors = ['SBM-FrequencyErrMask', 'PF-FrequencyErrMask', 'COE-FrequencyErrMask', 'ROE-FrequencyErrMask']
data_freq_box_error = []
for snr in snr_values:
    for column in freq_box_errors:
        data_freq_box_error.append(git(df, column, snr))
plot_box(data_freq_box_error)


mask_errors = ['SBM-MatchErrMask', 'PF-MatchErrMask', 'COE-MatchErrMask', 'ROE-MatchErrMask']
data_mask_error = []
for snr in snr_values:
    for column in mask_errors:
        data_mask_error.append(git(df, column, snr))
plot_box(data_mask_error)


time_mask_errors = ['SBM-TimeErrMask', 'PF-TimeErrMask', 'COE-TimeErrMask', 'ROE-TimeErrMask']
data_time_mask_error = []
for snr in snr_values:
    for column in time_mask_errors:
        data_time_mask_error.append(git(df, column, snr))
plot_box(data_time_mask_error)


freq_mask_errors = ['SBM-FrequencyErrMask', 'PF-FrequencyErrMask', 'COE-FrequencyErrMask', 'ROE-FrequencyErrMask']
data_freq_mask_error = []
for snr in snr_values:
    for column in freq_mask_errors:
        data_freq_mask_error.append(git(df, column, snr))
plot_box(data_freq_mask_error)

