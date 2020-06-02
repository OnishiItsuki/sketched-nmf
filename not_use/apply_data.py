import functionfile as ff
import time_measurement as tm
import pivoting_qr_q_eval as qr_eval
import v_error_eval as v_eval
import least_square_w_eval as ls_eval
import image_processing_function as ipf
import pandas as pd
import os
import numpy as np
import sys
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def apply_data(v, r, approximate_size, iteration, program_num, v_seed, wh_seed, c_mode, mode, w_path, nmfqp,
               column_sketching=True, convergence_D_mode=0, CDC=0):
    n, m = v.shape

    program_code = "realdata_" + ff.program_name(mode, c_mode, n, m, r, approximate_size, iteration, v_seed, wh_seed,
                                                 program_num, convergence_D_mode=convergence_D_mode, CDC=CDC)
    seeds = ff.two_seeds(wh_seed)

    print("-----------------   START   ------------------")
    print("start  " + program_code + "  >>>>>\n")
    print("seed of V=" + str(v_seed) + "  seed of WH=" + str(wh_seed))

    # calculate ----------------------------
    if mode == 0:
        t_result, w, h, w_s, h_os = tm.time_measurement(n, m, r, approximate_size, v, iteration, seeds, c_mode, column_sketching)
    elif mode == 1:
        nmf_error, _, w, h, w_s, h_os = qr_eval.pivoting_qr_q_eval(n, m, r, approximate_size, v, iteration, seeds, c_mode, column_sketching)
    elif mode == 2:
        nmf_error, snmf_error, w, h, w_s, h_os = v_eval.v_error_eval(n, m, r, approximate_size, v, iteration, seeds, c_mode, nmfqp, column_sketching)
    elif mode == 3:
        nmf_error, _, w, h, w_s, h_os = ls_eval.least_square_w_eval(n, m, r, approximate_size, v, iteration, seeds, c_mode, column_sketching)

    # result----------------------------
    # make directory
    os.makedirs(w_path + "/graph", exist_ok=True)
    os.makedirs(w_path + "/error", exist_ok=True)
    os.makedirs(w_path + "/time", exist_ok=True)
    os.makedirs(w_path + "/matrix", exist_ok=True)

    # plot error graph
    # if mode != 0:
    #     plt.plot(range(1, iteration + 1), nmf_error, label="existing method")
    #     if mode == 2:
    #         plt.plot(range(1, iteration + 1), snmf_error, label="proposed method")
    #     plt.xlabel("number of iteration")
    #     plt.ylabel("Frobenius norm")
    #     # plt.title("error plot")
    #     plt.legend()
    #     plt.savefig(w_path + "/graph/" + program_code + ".pdf")
    #     plt.close()
    #
    # # plot error difference graph
    # if mode == 2:
    #     plt.plot(range(1, iteration + 1), snmf_error - nmf_error)
    #     plt.xlabel("the number of iteration")
    #     plt.ylabel("error difference")
    #     # plt.title("The difference between Sketching NMF error and NMF error")
    #     plt.savefig(w_path + "/graph/Dif" + program_code + ".pdf")
    #     plt.close()

    # save error list
    if mode == 2:
        e_result = pd.DataFrame([nmf_error, snmf_error], index=["NMF error", "SNMF error"])
        e_result.T.to_csv(w_path + "/error/" + program_code + "_error.csv")
    elif mode != 0:
        e_result = pd.DataFrame([nmf_error], index=["NMF error"])
        e_result.T.to_csv(w_path + "/error/" + program_code + "_error.csv")

    # save time result
    # if mode == 0:
    #     tf_result = pd.DataFrame({"NMF time": t_result[0], "SNMF time": t_result[1]}, index=[0])
    #     tf_result.to_csv(w_path + "/time/" + program_code + "_time.csv")

    # save matrix
    # np.savetxt(w_path + "/matrix/w_" + program_code + ".csv", w, delimiter=",")
    # np.savetxt(w_path + "/matrix/h_" + program_code + ".csv", h, delimiter=",")
    # np.savetxt(w_path + "/matrix/W_s_" + program_code + ".csv", w_s, delimiter=",")
    # np.savetxt(w_path + "/matrix/H_s_" + program_code + ".csv", h_os, delimiter=",")
    print("\nfinish  " + program_code + "  >>>>>\n")

    if mode == 0:
        return t_result
    else:
        return nmf_error, snmf_error
