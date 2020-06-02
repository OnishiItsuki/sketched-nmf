import functionfile as ff
import time_measurement as tm
import pivoting_qr_q_eval as qr_eval
import v_error_eval as v_eval
import least_square_w_eval as ls_eval
import image_processing_function as ipf
import pandas as pd
import matplotlib
import os
import cv2
import numpy as np

import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.colors import Normalize

# chage to correct csv -------------------------------------------------------------------------------------------------
# r_path = "../mnt/workspace/sketchingNMF/CBCL/train/3(correction k,r)/MU"
# r_file_name = "SNMF_time(column_r, row_appr_size)"
# w_path = "../mnt/workspace/sketchingNMF/CBCL/train/3(correction k,r)/MU"
# w_file_name = "SNMF_time(column_r, row_appr_size)"
#
# M = np.loadtxt(r_path + "/" + r_file_name + ".csv")
# np.savetxt(w_path + "/" + w_file_name + ".csv", M, delimiter=",")

# add header -----------------------------------------------------------------------------------------------------------
# r_path = "../mnt/workspace/sketchingNMF/CBCL/train/3(correction k,r)/MU"
# r_file_name = "error_ave_SNMF(ave of -50:)"
# w_path = "../mnt/workspace/sketchingNMF/CBCL/train/3(correction k,r)/MU"
# w_file_name = "error_ave_SNMF(ave of -50:)"
#
# M = np.loadtxt(r_path + "/" + r_file_name + ".csv", delimiter=",")
# df = pd.DataFrame({"k=100": M[:, 0], "k=200": M[:, 1], "k=300": M[:, 2], "k=400": M[:, 3], "k=500": M[:, 4],
#                    "k=600": M[:, 5], "k=700": M[:, 6], "k=800": M[:, 7], "k=900": M[:, 8], "k=1000": M[:, 9]},
#                   index=["r=5", "r=10", "r=15", "r=20", "r=25", "r=30", "r=35", "r=40", "r=45", "r=50"])
# df.to_csv(w_path + "/" + w_file_name + ".csv")

# insert ---------------------------------------------------------------------------------------------------------------
# insert_p = w_file_name.find("_realdata_")
# w_file_name = w_file_name[:insert_p] + "_" + str(image_number) + w_file_name[insert_p:]

# make table  ----------------------------------------------------------------------------------------------------------
def checkerboard_table(data, w_path, w_file_name, fmt='{:.2f}'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = data.shape
    width, height = 1.2 / ncols, 1 / nrows

    tb.auto_set_font_size(False)
    tb.set_fontsize(0.0001)
    for (i, j), val in np.ndenumerate(data):
        tb.add_cell(i, j, width, height, text=fmt.format(val), loc='center')

    # row label
    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text=label, loc='right', edgecolor='none')

    # col label
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', edgecolor='none')
    ax.add_table(tb)
    plt.subplots_adjust(left=0.15, right=0.985, bottom=0.3, top=0.6)
    # plt.subplots_adjust(left=0.15, right=0.985, bottom=0.02, top=0.84)

    # fig, ax = plt.subplots(figsize=(300, 300))
    # ax.axis('off')
    # ax.axis('tight')
    # ax.table(cellText=data.values, colLabels=data.columns, rowLabels=data.index, loc='center', bbox=[0, 0, 1, 1])
    plt.savefig(w_path + w_file_name)