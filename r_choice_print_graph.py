import image_processing_function as ipf
import numpy as np
import sys
import os

# 1: cbcl,yale  2-: test,bic,ls,bcv,all_r

ite = 1500
row_sep = 5
col_sep = 5
tmp = sys.argv

# preparation  ---------------------------------------------------------------------------------------------------------
all_r_flag = False
if "all_r" in tmp[2:]:
    r_list = range(5, 101, 5)
    all_r_flag = True
if "cbcl" in tmp[1]:
    Use_data = "CBCL/train"
    seed_list = range(1, 11)
    title = "CBCL Face Database"
elif "yale" in tmp[1]:
    Use_data = "YaleFD/faces"
    seed_list = range(1, 11)
    title = "The Yale Face Database"

use_method = []
if "all" in tmp[2:]:
    use_method.extend(["BIC", "LS", "BCV"])
if "bic" in tmp[2:]:
    use_method.append("BIC")
if "ls" in tmp[2:]:
    use_method.append("LS")
if "bcv" in tmp[2:]:
    use_method.append("BCV")

base_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/r_choice".format(Use_data)
write_path = "{}/plot_result".format(base_path)
if "test" in tmp[2:]:
    seed_list = [1]
    write_path = "{}/test_plot_result".format(base_path)
os.makedirs(write_path, exist_ok=True)

# plot  ----------------------------------------------------------------------------------------------------------------
if "BIC" in use_method:
    if not all_r_flag:
        if tmp[1] == "cbcl":
            r_list = range(20, 101, 5)
        elif tmp[1] == "yale":
            r_list = range(5, 51, 5)

    bic = np.zeros([len(seed_list), len(r_list)])
    for r_i, r in enumerate(r_list):
        for seed_i, seed in enumerate(seed_list):
            bic[seed_i, r_i] = np.loadtxt("{}/BIC/r{}_seed{}.csv".format(base_path, r, seed), delimiter=",").reshape(1)
    ipf.box_graph_plot(bic, "{}/BIC(r{}_{},seed{}_{})".format(write_path, r_list[0], r_list[-1], seed_list[0], seed_list[-1]),
                       x_tuple=r_list, x_label="r", y_label="BIC")

if "LS" in use_method:
    ls = np.zeros([len(seed_list), len(r_list)])
    for r_i, r in enumerate(r_list):
        for seed_i, seed in enumerate(seed_list):
            ls[seed_i, r_i] = np.loadtxt("{}/LS/r{}_seed{}.csv".format(base_path, r, seed), delimiter=",")[-1]
    ipf.box_graph_plot(ls, "{}/LS(r{}_{},seed{}_{})".format(write_path, r_list[0], r_list[-1], seed_list[0], seed_list[-1]),
                       x_tuple=r_list, x_label="r", y_label="Least Square error")

if "BCV" in use_method:
    if not all_r_flag:
        if tmp[1] == "cbcl":
            r_list = range(35, 101, 5)
        elif tmp[1] == "yale":
            r_list = range(5, 51, 5)

    bcv = np.zeros([len(seed_list), len(r_list)])
    for r_i, r in enumerate(r_list):
        for seed_i, seed in enumerate(seed_list):
            bcv[seed_i, r_i] = np.loadtxt("{}/BCV_sep({},{})/r{}_seed{}.csv"
                                          .format(base_path, row_sep, col_sep, r, seed), delimiter=",").reshape(1)
    ipf.box_graph_plot(bcv, "{}/BCV(r{}_{},seed{}_{},sep{}_{})"
                   .format(write_path, r_list[0], r_list[-1], seed_list[0], seed_list[-1], row_sep, col_sep),
                       x_tuple=r_list, x_label="r", y_label="BCV")
