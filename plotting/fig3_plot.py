import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd

# df = pd.read_csv('./fig3_data/analysis3-result-brownnoise.csv')
# df = pd.read_csv('./fig3_data/analysis3-result-eeg.csv')
df = pd.read_csv('./fig3_data/analysis3-result-pinknoise.csv')
print(df)

# dtip = "BROWN"
# dtip = "EEG"
dtip = "PINK"
# trans = "CWT"
# trans = "SLT"
trans = "STFT"


TESTS = 4

snr_values = [0.1, 0.25, 0.5, 1, 2]
box_colors = ['darkkhaki', 'royalblue', 'pink', 'lightgreen']
weights = ['bold', 'semibold', 'light', 'heavy']




def get_data(df, column, snr):
    # print(f"-------------{snr} -> {column}-------------")
    # print(df.query(f"SNR == {snr}")[column])
    # print()
    return df.query(f"SNR == {snr}")[column].to_numpy()


def get_restricted_data(df, column, snr, check_col):
    df = df.copy()
    # print(f"-------------{snr} -> {column}-------------")
    # print(df.query(f"SNR == {snr} and {str(column)} < 1")[column])
    # print()

    return df.query(f"SNR == {snr} and {str(check_col)} < 1")[column].to_numpy(), df.query(f"SNR == {snr} and {str(check_col)} == 1")[column].count()
    # return df.query(f"SNR == {snr} ")[column].to_numpy(), df.query(f"SNR == {snr} and {str(check_col)} == 1")[column].count()







def plot_box(dtip, trans, title, data, METHODS):
    data = data.copy()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # fig.canvas.manager.set_window_title('A Boxplot Example')
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
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel='SNR',
        ylabel='Value',
    )

    # Now fill the boxes with desired colors
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):

        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5): # 5=box coordinates
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])

        box_coords = np.column_stack([box_x, box_y])

        # Alternate among colors
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % len(METHODS)]))

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)

    # top = max([np.amax(x)+np.std(x) for x in data if x.size != 0])
    # bottom = min([np.amin(x)-np.std(x) for x in data if x.size != 0])
    # ax1.set_ylim(bottom, top)
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
    plt.savefig(f"fig3/fig3 - {dtip} - {trans} - {title}.svg")
    plt.savefig(f"fig3/fig3 - {dtip} - {trans} - {title}.png")
    plt.show()



def bar_plot(dtip, trans, title, bar_plot_values, METHODS):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    name_shit = []
    for snr in snr_values:
        for nime in METHODS:
            name_shit.append(nime+f"-SNR{snr}")
    print(name_shit)
    if len(METHODS) == 3:
        plt.bar(name_shit, bar_plot_values, color=['darkkhaki', 'royalblue', 'pink'], width = 0.5)
    if len(METHODS) == 2:
        plt.bar(name_shit, bar_plot_values, color=['darkkhaki', 'royalblue'], width = 0.5)
    plt.xticks(rotation='vertical')
    plt.savefig(f"fig3/fig3 - {dtip} - {trans} - {title}.svg")
    plt.savefig(f"fig3/fig3 - {dtip} - {trans} - {title}.png")
    plt.show()




bar_plot_values = []
box_errors = [f'{trans}BoxSBMMatchErr', f'{trans}BoxPFMatchErr', f'{trans}BoxOEMatchErr']
data_box_error = []
for snr in snr_values:
    for column, check_col in zip(box_errors, box_errors):
        # test = git(df, column, snr)
        test, number = get_restricted_data(df, column, snr, check_col)
        # print(number)
        bar_plot_values.append(number/200)
        data_box_error.append(test)
plot_box(dtip, trans, "Box Match Error", data_box_error, box_errors)
bar_plot(dtip, trans, "Box Misses", bar_plot_values, box_errors)


time_box_errors = [f'{trans}BoxSBMTimeErr', f'{trans}BoxPFTimeErr', f'{trans}BoxOETimeErr']
data_time_box_error = []
for snr in snr_values:
    for column, check_col in zip(time_box_errors, box_errors):
        test, number = get_restricted_data(df, column, snr, check_col)
        data_time_box_error.append(test)
plot_box(dtip, trans, "Box Time Error", data_time_box_error, time_box_errors)


freq_box_errors = [f'{trans}BoxSBMFreqErr', f'{trans}BoxPFFreqErr', f'{trans}BoxOEFreqErr']
data_freq_box_error = []
for snr in snr_values:
    for column, check_col in zip(freq_box_errors, box_errors):
        test, number = get_restricted_data(df, column, snr, check_col)
        data_freq_box_error.append(test)
plot_box(dtip, trans, "Box Freq Error", data_freq_box_error, freq_box_errors)







bar_plot_values = []
mask_errors = [f'{trans}MaskSBMMatchErr', f'{trans}MaskPFMatchErr']
data_mask_error = []
for snr in snr_values:
    for column, check_col in zip(mask_errors, mask_errors):
        test, number = get_restricted_data(df, column, snr, check_col)
        # print(number)
        bar_plot_values.append(number/200)
        data_mask_error.append(test)
plot_box(dtip, trans, "Mask Match Error", data_mask_error, mask_errors)
bar_plot(dtip, trans, "Mask Misses", bar_plot_values, mask_errors)





time_mask_errors = [f'{trans}MaskSBMTimeErr', f'{trans}MaskPFTimeErr']
data_time_mask_error = []
for snr in snr_values:
    for column, check_col in zip(time_mask_errors, mask_errors):
        test, number = get_restricted_data(df, column, snr, check_col)
        data_time_mask_error.append(test)
plot_box(dtip, trans, "Mask Time Error", data_time_mask_error, time_mask_errors)


freq_mask_errors = [f'{trans}MaskSBMFreqErr', f'{trans}MaskPFFreqErr']
data_freq_mask_error = []
for snr in snr_values:
    for column, check_col in zip(freq_mask_errors, mask_errors):
        test, number = get_restricted_data(df, column, snr, check_col)
        data_freq_mask_error.append(test)
plot_box(dtip, trans, "Mask Freq Error", data_freq_mask_error, freq_mask_errors)



