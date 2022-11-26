import numpy as np


def find_extrema_1D(vect, stride):
    mins = np.zeros_like(vect)
    maxs = np.zeros_like(vect)

    last_index = 0
    current_index = 0
    current_difference = 0
    last_difference = 0
    while (current_index+stride < len(vect)):
        last_difference = vect[current_index+stride] - vect[current_index]

        if last_difference == 0:
            current_index+=1
            continue

        index_offset = (current_index + last_index) // 2

        if current_difference > 0 and last_difference < 0:
            maxs[index_offset] = 1
        elif current_difference < 0 and last_difference > 0:
                mins[index_offset] = 1

        current_difference = last_difference
        current_index += 1
        last_index = current_index

    return mins.astype(bool), maxs.astype(bool)


def find_extrema_2D_row(data, stride):
    #rows
    mat_mins_by_row = []
    mat_maxs_by_row = []
    for i in range(len(data)):
        row_mins, row_maxs = find_extrema_1D(data[i], stride)
        mat_mins_by_row.append(row_mins.tolist())
        mat_maxs_by_row.append(row_maxs.tolist())

    mat_mins_by_row = np.array(mat_mins_by_row).astype(bool)
    mat_maxs_by_row = np.array(mat_maxs_by_row).astype(bool)

    return mat_mins_by_row, mat_maxs_by_row


def find_extrema_2D(data, stride, type):
    if isinstance(stride, int):
        stride_x = stride_y = stride
    elif isinstance(stride, tuple) or isinstance(stride, list):
        stride_x = stride[0]
        stride_y = stride[1]

    #rows
    mat_mins_by_row = []
    mat_maxs_by_row = []
    for i in range(len(data)):
        row_mins, row_maxs = find_extrema_1D(data[i], stride_x)
        mat_mins_by_row.append(row_mins.tolist())
        mat_maxs_by_row.append(row_maxs.tolist())

    mat_mins_by_row = np.array(mat_mins_by_row)
    mat_maxs_by_row = np.array(mat_maxs_by_row)

    # cols
    mat_mins_by_cols = []
    mat_maxs_by_cols = []
    for i in range(len(data[0])):
        col_mins, col_maxs = find_extrema_1D(data[:, i], stride_y)
        mat_mins_by_cols.append(col_mins.tolist())
        mat_maxs_by_cols.append(col_maxs.tolist())

    mat_mins_by_cols = np.array(mat_mins_by_cols).T
    mat_maxs_by_cols = np.array(mat_maxs_by_cols).T

    if type == 'or':
        mat_mins = np.logical_or(mat_mins_by_row, mat_mins_by_cols).astype(bool)
        mat_maxs = np.logical_or(mat_maxs_by_row, mat_maxs_by_cols).astype(bool)
    if type == 'and':
        mat_mins = np.logical_and(mat_mins_by_row, mat_mins_by_cols).astype(bool)
        mat_maxs = np.logical_and(mat_maxs_by_row, mat_maxs_by_cols).astype(bool)

    return mat_mins, mat_maxs