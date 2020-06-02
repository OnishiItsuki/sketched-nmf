import pandas as pd
import sys
import numpy as np
import cv2
import os
# import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter


def read_pgm(path):
    path = path + "/"
    f_list = os.listdir(path)
    sample = cv2.imread(path + f_list[0], 0)
    elm_size = sample.shape[0] * sample.shape[1]
    v = np.zeros((elm_size, len(f_list)))

    for i, f_name in enumerate(f_list):
        v[:, i] = cv2.imread(path + f_name, 0).reshape([elm_size])

    return v, sample.shape[0], sample.shape[1]


def save_sample(v, im_var, im_hol, path, filename, program_num=None, image_num=0):
    if program_num is not None:
        path = path + "/" + program_num
    os.makedirs(path, exist_ok=True)

    plt.subplot(111)
    plt.imshow(v[:, image_num].reshape([im_var, im_hol]), cmap="gray")
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)  # not plot label
    plt.tick_params(bottom=False, left=False, right=False, top=False)  # not plot line
    plt.savefig("{}/{}.png".format(path, filename), bbox_inches="tight", pad_inches=0.0, dpi=1000)


def save_original_image(r_path, w_path, w_file_name, image_num_list):
    tmp_v, im_var, im_hol = read_pgm(r_path)
    v = np.zeros([im_var, im_hol, len(image_num_list)])
    for im_i, im_num in enumerate(image_num_list):
        save_sample(tmp_v, im_var, im_hol, w_path, im_num, image_num=im_num)
        v[:, :, im_i] = np.reshape(tmp_v[:, im_num], [im_var, im_hol])
    save_four_block_sample(v, w_path, w_file_name, "Ground Truth")


def save_w_image(r_path, r_file_name, w_path, image_size, image_number):
    m = np.loadtxt(r_path + "/" + r_file_name + ".csv", delimiter=",")
    save_sample(m, image_size[0], image_size[1], w_path, str(image_number), image_num=image_number)


def calculate_v_then_save_sample(r_path, r_file_name, w_path, image_size, image_number, snmf=True):
    v = calculate_v(r_path, r_file_name, snmf)
    if snmf:
        save_sample(v, image_size[0], image_size[1], w_path, "snmf_" + str(image_number), image_num=image_number)
    else:
        save_sample(v, image_size[0], image_size[1], w_path, "nmf_" + str(image_number), image_num=image_number)


def calculate_v(r_path, r_file_name, snmf=True, t_flag=False):
    if snmf:
        w = "W_s_"
        h = "H_s_"
    else:
        w = "w_"
        h = "h_"

    w = np.loadtxt(r_path + w + r_file_name + ".csv", delimiter=",")
    h = np.loadtxt(r_path + h + r_file_name + ".csv", delimiter=",")
    v = np.dot(w, h)
    if t_flag:
        return v.T
    else:
        return v


def read_wh(r_path, r_file_name, snmf=True):
    if snmf:
        w = "W_s_"
        h = "H_s_"
    else:
        w = "w_"
        h = "h_"

    w = np.loadtxt(r_path + w + r_file_name + ".csv", delimiter=",")
    h = np.loadtxt(r_path + h + r_file_name + ".csv", delimiter=",")
    return w, h


def mse_calculate(x, y):
    if (x.shape[0] != y.shape[0]) | (x.shape[1] != y.shape[1]):
        print("Error : Matrix size is not matching", file=sys.stderr)
        sys.exit(1)
    element_size = x.shape[0] * x.shape[1]
    return np.sqrt(np.linalg.norm(x - y) ** 2 / element_size)
    # return np.sqrt(np.linalg.norm(x - y) ** 2 / element_size), np.sqrt(np.max(np.linalg.norm(x - y) ** 2))


def make_mse_dataframe(mse_result, ap_list):
    mse_result = np.mean(mse_result, axis=1)

    df_mse = pd.DataFrame({"k=" + str(ap_list[0]): mse_result[0, 0]}, index=["GT-NMF"])
    df_mse.loc["GT-SNMF"] = mse_result[0, 1]
    df_mse.loc["NMF-SNMF"] = mse_result[0, 2]

    for ap_i, ap_s in enumerate(ap_list[1:]):
        df_mse["k=" + str(ap_s)] = mse_result[ap_i + 1, :]
    return df_mse


def save_four_block_sample(images, w_path, w_file_name, title):
    os.makedirs(w_path + "PNG/", exist_ok=True)

    plt.subplots_adjust(wspace=0.05, hspace=0.01)
    if "CBCL" in w_path:
        for im_i, im_num in enumerate([1, 2, 4, 5]):
            plt.subplot(2, 3, im_num)
            plt.imshow(images[:, :, im_i], cmap="gray")
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)  # not plot label
            plt.tick_params(bottom=False, left=False, right=False, top=False)  # not plot line
        plt.savefig(w_path + "PNG/" + w_file_name + ".png", bbox_inches="tight", pad_inches=0.0, dpi=1000)
        # plt.suptitle(title, size=23, x=0.38, y=0.93)
    elif "Yale" in w_path:
        for im_i, im_num in enumerate([1, 2, 3, 4]):
            plt.subplot(2, 2, im_num)
            plt.imshow(images[:, :, im_i], cmap="gray")
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)  # not plot label
            plt.tick_params(bottom=False, left=False, right=False, top=False)  # not plot line
        plt.savefig(w_path + "PNG/" + w_file_name + ".png", bbox_inches="tight", pad_inches=0.0, dpi=1000)
        # plt.suptitle(title, size=25, x=0.52, y=0.96)

    plt.savefig(w_path + w_file_name + ".pdf", bbox_inches="tight", pad_inches=0.0, dpi=1000)
    plt.close()


def save_error_map(error_map, bar_max, data_num, r, ap_num, w_path, color_bar, title_off=False):
    os.makedirs(w_path + "png/", exist_ok=True)

    error_map = np.abs(error_map)

    if data_num == 0:
        title = "Ground Truth - Existing method"
        s_title = "GT_NMF"
    elif data_num == 1:
        title = "Ground Truth - Proposed method"
        s_title = "GT_SNMF"
    elif data_num == 2:
        title = "Existing method - Proposed method"
        s_title = "NMF_SNMF"

    plt.subplots_adjust(wspace=0.05, hspace=0.01)
    if "CBCL" in w_path:
        x, y, f_size = [0.38, 0.93, 20]
        for im_i, im_num in enumerate([1, 2, 4, 5]):
            plt.subplot(2, 3, im_num)
            plt.imshow(error_map[:, :, im_i], norm=Normalize(vmin=0, vmax=bar_max))
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)  # not plot label
            plt.tick_params(bottom=False, left=False, right=False, top=False)  # not plot line
    elif "Yale" in w_path:
        x, y, f_size = [0.52, 0.95, 25]
        for im_i, im_num in enumerate([1, 2, 3, 4]):
            plt.subplot(2, 2, im_num)
            plt.imshow(error_map[:, :, im_i], norm=Normalize(vmin=0, vmax=bar_max))
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)  # not plot label
            plt.tick_params(bottom=False, left=False, right=False, top=False)  # not plot line

    if color_bar:
        w_file_name = "r={},ap={}_error_map({})".format(r, ap_num, s_title)
        if "CBCL" in w_path:
            cax = plt.axes([0.67, 0.13, 0.02, 0.72])
        elif "Yale" in w_path:
            cax = plt.axes([0.93, 0.13, 0.02, 0.73])
        cbar = plt.colorbar(cax=cax)
        plt.savefig("{}png/{}.png".format(w_path, w_file_name), bbox_inches="tight", pad_inches=0.0, dpi=1000)
        cbar.set_label('error')
    else:
        w_file_name = "no_color_bar_r={},ap={}_error_map({})".format(r, ap_num, s_title)
        plt.savefig("{}png/{}.png".format(w_path, w_file_name), bbox_inches="tight", pad_inches=0.0, dpi=1000)
    if not title_off:
        # plt.suptitle(title, size=f_size, x=x, y=y)
        a = 1
    plt.savefig(w_path + w_file_name + ".pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()


def box_graph_plot(data_matrix, w_path, line=None, x_tuple=None, x_label="", y_label="", title="", y_lim=None):
    plt.rcParams['font.size'] = 20
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.15)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    if np.max(data_matrix) >= 100:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if data_matrix.shape[0] == 1:
        data_matrix = data_matrix.reshape(data_matrix.shape[1])
        ax.scatter(range(len(data_matrix)), data_matrix, marker='o')
    else:
        ax.boxplot(data_matrix, whis="range")

    ax.set_xticklabels(x_tuple)
    if line is not None:
        plt.hlines(line, 0, data_matrix.shape[1] + 1, "blue", linestyles='dashed', label="exiting method")
        plt.legend()
    if y_lim is not None:
        plt.ylim(y_lim)

    plt.savefig("{}.png".format(w_path), dpi=1000)
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    # plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig("{}.pdf".format(w_path))
    plt.close()


def ite_cost_plot(nmf_error, snmf_error, ite, write_path, program_code):
    plt.rcParams['font.size'] = 20
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams["legend.markerscale"] = 2
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.edgecolor"] = 'black'
    plt.rcParams["legend.framealpha"] = 1

    os.makedirs(write_path + "png/", exist_ok=True)
    # plot error graph
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    if (np.max(nmf_error) >= 100) | (np.max(snmf_error) >= 100):
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.plot(range(1, ite + 1), nmf_error[:ite], label="existing method")
    plt.plot(range(1, ite + 1), snmf_error[:ite], label="proposed method")
    plt.savefig(write_path + "png/" + program_code + ".png", dpi=1000)

    fig.subplots_adjust(left=0.15, bottom=0.15)
    plt.xlabel("iterations")
    if "W" in program_code:
        plt.ylabel("Frobenius norm error of W")
    else:
        plt.ylabel("Reconstruction error")
    # if "MU" in write_path:
    #     plt.title("MU")
    # elif "HALS" in write_path:
    #     plt.title("HALS")
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(write_path + program_code + ".pdf")
    plt.close()

    # plot error difference graph
    # plt.plot(range(1, ite + 1), snmf_error - nmf_error)
    # plt.xlabel("umber of iterations")
    # plt.ylabel("error difference")
    # # plt.title("The difference between Sketching NMF and NMF error")
    # plt.savefig("/graph/Dif{}.pdf".format(write_path, write_file_name))
    # plt.close()


def multi_line_graph_plot(data_matrix, x, legend_label, w_path, line=None, x_label="", y_label="", title="", y_lim=None):
    # row: data  column: sample
    plt.rcParams['font.size'] = 20
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    if np.max(data_matrix) >= 100:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for i in range(data_matrix.shape[1]):
        plt.plot(x, data_matrix[:, i], label=legend_label[i])

    if line is not None:
        plt.hlines([line], np.min(x), np.max(x), "blue", linestyles='dashed')
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.savefig("{}.png".format(w_path), dpi=1000)

    fig.subplots_adjust(left=0.15, bottom=0.15)
    # plt.title(title)
    plt.legend(fontsize=15)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig("{}.pdf".format(w_path))
    plt.close()
