import functionfile as ff
import image_processing_function as ipf
import numpy as np
import pandas as pd
import sys
import os
import sys

# 1: cbcl,yale,random  2: mu,hals,fgd,gcd,all  3-: test,vnorm,wnorm,ite_time,k_time,mse

# check parameter !!!!!!!!!!!!!!!!!!!!


def cal_three_mse(r_list, ap_list, seed_list, iteration, use_data, program_num, c_method):
    result = np.zeros([len(seed_list), len(ap_list), 3])  # [0]:ori-nmf  [1]:ori-snmf  [2]:nmf-snmf

    v_original, im_var, im_hol = ipf.read_pgm("/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + use_data)
    v_list = np.zeros([v_original.shape[0], v_original.shape[1], 3])  # [0]:original  [1]:snmf  [2]:nmf
    v_list[:, :, 0] = v_original

    for ap_i, ap in enumerate(ap_list):
        for seed_i, seed in enumerate(seed_list):
            path, file_name = ff.path_and_file_name(r_list[0], ap, iteration, seed, use_data, program_num, c_method,
                                                    "r,k", directory="matrix")
            v_list[:, :, 1] = ipf.calculate_v(path, file_name, snmf=True)
            path, file_name = ff.path_and_file_name(r_list[0], ap_list[-1], iteration, seed, use_data, program_num, c_method,
                                                    "r,k", directory="matrix")
            v_list[:, :, 2] = ipf.calculate_v(path, file_name, snmf=False)
            for i in range(3):
                result[seed_i, ap_i, (i + 1) % 3] = ipf.mse_calculate(v_list[:, :, i], v_list[:, :, (i + 1) % 3])
    return result


if __name__ == "__main__":
    n = 77760  # if you don't need to enter specific values, you must set n=0. Using to make path.
    m = 165  # if you don't need to enter specific values, you must set m=0. Using to make path.

    V_seed = 0
    tmp = sys.argv
    # seed_list = [1]
    seed_list = range(1, 11)

    k_list = [1000]
    # k_list = [1500]

    ite_list = []  # use in ite_time
    for i in range(100, 1001, 100):
        ite_list.append(i)
    # ite_time_k_list = [2500, 7500, 30000, 77760]  # use in ite_time
    ite_time_k_list = [10000]  # use in ite_time

    # preparation  -----------------------------------------------------------------------------------------------------
    C_method_list = []
    if tmp[2] == "mu":
        C_method_list.append("MU")
    elif tmp[2] == "hals":
        C_method_list.append("HALS")
    elif tmp[2] == "fgd":
        C_method_list.append("FGD")
    elif tmp[2] == "gcd":
        C_method_list.append("GCD")
    elif tmp[2] == "all":
        C_method_list.extend(["MU", "HALS", "FGD", "GCD"])
    else:
        print("box_plot.py Error: {} is not NMF algorithm".format(tmp[2]), file=sys.stderr)
        sys.exit(1)

    if tmp[1] == "cbcl":
        Use_data = "CBCL/train"
        r_list = [50]
        if len(k_list) == 0:
            k_list = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2429]
    elif tmp[1] == "yale":
        Use_data = "YaleFD/faces"
        r_list = [25]
        # r_list = [50]
        if len(k_list) == 0:
            k_list = [1000, 2500, 5000, 7500, 10000, 30000, 50000, 77760]
    elif tmp[1] == "random":
        pass
    else:
        print("Error: {} is not face database".format(tmp[1]), file=sys.stderr)
        sys.exit(1)
    use_method = []
    if tmp[1] == "random":
        Use_data = "random_matrix"
        r_list = [50]  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        head = ""
        # Program_num = "17(for_MIRU)"  # change code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Program_num = "18(for_MIRU_FGD_bag_fix)"
        Program_num = "19(for_MIRU_not_transpose)"
        # Program_num = "ttest"
        if "all" in tmp[3:]:
            use_method.extend(["V_norm", "W_norm", "ite_time"])
    else:
        Program_num = "11(for_MIRU_FGD_bag_fix)"  # change code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        head = "realdata_"
        if "all" in tmp[3:]:
            use_method.extend(["k_time", "mse"])

    if "vnorm" in tmp[3:]:
        use_method.append("V_norm")
    if "wnorm" in tmp[3:]:
        use_method.append("W_norm")
    if "ite_time" in tmp[3:]:
        use_method.append("ite_time")
    if "k_time" in tmp[3:]:
        use_method.append("k_time")
    if "mse" in tmp[3:]:
        use_method.append("mse")

    # to reset ite_list and ite_time_k_list
    ite_list_set = False
    ite_time_k_list_set = False
    if len(ite_list) == 0:
        ite_list_set = True
    if len(ite_time_k_list) == 0:
        ite_time_k_list_set = True

    for C_method in C_method_list:
        if C_method == "MU":
            if ite_list_set:
                ite_list = [250, 500, 750]  # use in ite_time
                for i in range(1000, 5001, 500):
                    ite_list.append(i)
            if ite_time_k_list_set:
                ite_time_k_list = [500, 1000, 5000]  # use in ite_time
            if tmp[1] == "random":
                iteration = 5000
            else:
                iteration = 3000

        elif C_method == "HALS":
            if ite_list_set:
                ite_list = [50, 100, 200, 300, 400, 500, 600, 750]  # use in ite_time
            if ite_time_k_list_set:
                ite_time_k_list = [500, 2500, 5000]  # use in ite_time
            if tmp[1] == "random":
                iteration = 400
            else:
                iteration = 400

        elif C_method == "FGD":
            if ite_list_set:
                ite_list = []  # use in ite_time
                for i in range(50, 1001, 50):
                    ite_list.append(i)
            if ite_time_k_list_set:
                ite_time_k_list = [1000, 1500, 2000]  # use in ite_time
            if tmp[1] == "random":
                iteration = 1000
            else:
                iteration = 1000

        elif C_method == "GCD":
            if ite_list_set:
                ite_list = []  # use in ite_time
                for i in range(50, 1001, 50):
                    ite_list.append(i)
            if ite_time_k_list_set:
                ite_time_k_list = [1000, 1500, 2000]  # use in ite_time
            if tmp[1] == "random":
                iteration = 1000
            else:
                iteration = 400

        write_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/box_plot_result".format(Use_data, Program_num, C_method)
        test_flag = False
        if "test" in tmp[3:]:
            # seed_list = [1]
            test_flag = True
            write_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/test_box_plot_result/{}".format(Use_data, Program_num, C_method)

        os.makedirs(write_path, exist_ok=True)

        # plot  --------------------------------------------------------------------------------------------------------
        print("\nstart  {}  {}  {}  >>>>>>>>>\n".format(Use_data, C_method, use_method))
        for r in r_list:
            if "V_norm" in use_method:
                result = np.zeros([len(seed_list), len(k_list)])
                nmf_tmp = np.zeros([len(seed_list)])
                for k_i, k in enumerate(k_list):
                    for seed_i, seed in enumerate(seed_list):
                        nmf, snmf = ff.read_cost_func_error(r, k, iteration, seed, Use_data, Program_num, C_method, test_flag, n=n, m=m, v_or_w="v")
                        result[seed_i, k_i] = snmf[iteration - 1]
                        nmf_tmp[seed_i] = nmf[iteration - 1] if k == k_list[-1] else None
                line_num = np.mean(nmf_tmp, axis=0)
                if tmp[1] == "radnom":
                    ipf.box_graph_plot(result, "{}/ap_V(k{}_{},seed{}_{},n{},m{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], n, m, r, iteration),
                                       x_tuple=k_list, x_label="k", y_label="Reconstruction error", line=line_num, title=C_method)
                else:
                    ipf.box_graph_plot(result, "{}/ap_V(k{}_{},seed{}_{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], r, iteration),
                                       x_tuple=k_list, x_label="k", y_label="Reconstruction error", line=line_num, title=C_method)
            if "W_norm" in use_method:
                result = np.zeros([len(seed_list), len(k_list)])
                nmf_tmp = np.zeros([len(seed_list)])
                for k_i, k in enumerate(k_list):
                    for seed_i, seed in enumerate(seed_list):
                        nmf, snmf = ff.read_cost_func_error(r, k, iteration, seed, Use_data, Program_num, C_method, test_flag, n=n, m=m, v_or_w="w")
                        result[seed_i, k_i] = snmf[iteration - 1]
                        nmf_tmp[seed_i] = nmf[iteration - 1] if k == k_list[-1] else None
                line_num = np.mean(nmf_tmp, axis=0)
                if tmp[1] == "radnom":
                    ipf.box_graph_plot(result, "{}/ap_W(k{}_{},seed{}_{},n{},m{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], n, m, r, iteration),
                                       x_tuple=k_list, x_label="k", y_label="Frobenius norm error of W", line=line_num, title=C_method)
                else:
                    ipf.box_graph_plot(result, "{}/ap_W(k{}_{},seed{}_{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], r, iteration),
                                       x_tuple=k_list, x_label="k", y_label="Frobenius norm error of W", line=line_num, title=C_method)

            if "mse" in use_method:
                os.makedirs(write_path + "/k_MSE", exist_ok=True)
                for k_i, k in enumerate(k_list):
                    mse = cal_three_mse(r_list, k_list, seed_list, iteration, Use_data, Program_num, C_method)
                    print("finish  calculate  MSE  k={}".format(k))
                nmf = np.mean(mse[:, :, 0], axis=1)
                nmf = np.mean(nmf)
                if tmp[1] == "random":
                    ipf.box_graph_plot(mse[:, :, 1], "{}/k_MSE/GT_SNMF(k{}_{},seed{}_{},n{},m{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], n, m, r, iteration),
                                       x_tuple=k_list, x_label="k", y_label="RMSE", line=nmf)
                    ipf.box_graph_plot(mse[:, :, 2], "{}/k_MSE/NMF_SNMF(k{}_{},seed{}_{},n{},m{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], n, m, r, iteration),
                                       x_tuple=k_list, x_label="k", y_label="RMSE")
                else:
                    ipf.box_graph_plot(mse[:, :, 1], "{}/k_MSE/GT_SNMF(k{}_{},seed{}_{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], r, iteration),
                                       x_tuple=k_list, x_label="k", y_label="RMSE", line=nmf)
                    ipf.box_graph_plot(mse[:, :, 2], "{}/k_MSE/NMF_SNMF(k{}_{},seed{}_{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], r, iteration),
                                       x_tuple=k_list, x_label="k", y_label="RMSE")

            if "k_time" in use_method:
                result = np.zeros([len(seed_list), len(k_list)])
                nmf_tmp = np.zeros([len(seed_list)])
                for k_i, k in enumerate(k_list):
                    for seed_i, seed in enumerate(seed_list):
                        nmf, snmf = ff.read_time(r, k, iteration, seed, Use_data, Program_num, C_method, test_flag, n=n, m=m)
                        result[seed_i, k_i] = snmf
                        nmf_tmp[seed_i] = nmf if k == k_list[-1] else None
                line_num = np.mean(nmf_tmp, axis=0)
                if tmp[1] == "random":
                    ipf.box_graph_plot(result, "{}/k_time(k{}_{},seed{}_{},n{},m{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], n, m, r, iteration),
                                       x_tuple=k_list, x_label="k", y_label="Time (sec)", line=line_num, title=C_method)
                else:
                    ipf.box_graph_plot(result, "{}/k_time(k{}_{},seed{}_{},r{},ite{})"
                                       .format(write_path, k_list[0], k_list[-1], seed_list[0], seed_list[-1], r,
                                               iteration),
                                       x_tuple=k_list, x_label="k", y_label="Time (sec)", line=line_num, title=C_method)

            if "ite_time" in use_method:
                result = np.zeros([len(seed_list), len(ite_list), len(ite_time_k_list) + 1])
                legend_list = ["existing method"]
                for k_i, k in enumerate(ite_time_k_list):
                    # lengend label !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    legend_list.append("proposed method k={}".format(k))
                    for ite_i, ite in enumerate(ite_list):
                        for seed_i, seed in enumerate(seed_list):
                            nmf, snmf = ff.read_time(r, k, ite, seed, Use_data, Program_num, C_method, test_flag, n=n, m=m)
                            result[seed_i, ite_i, k_i + 1] = snmf
                            result[seed_i, ite_i, 0] = nmf if k == ite_time_k_list[-1] else None
                result = np.mean(result, axis=0)
                if tmp[1] == "random":
                    ipf.multi_line_graph_plot(result, ite_list, legend_list, "{}/ite_time(ite{}_{},seed{}_{},n{},m{},r{},k{})"
                                              .format(write_path, ite_list[0], ite_list[-1], seed_list[0], seed_list[-1], n, m, r, ite_time_k_list).replace(" ", ""),
                                              x_label="iterations", y_label="Time (sec)", title=C_method)
                else:
                    ipf.multi_line_graph_plot(result, ite_list, legend_list, "{}/ite_time(ite{}_{},seed{}_{},r{},k{})"
                                              .format(write_path, ite_list[0], ite_list[-1], seed_list[0], seed_list[-1], r, ite_time_k_list).replace(" ", ""),
                                              x_label="iterations", y_label="Time (sec)", title=C_method)

