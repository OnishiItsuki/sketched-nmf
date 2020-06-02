import functionfile as ff
import time_measurement as tm
import image_processing_function as ipf
import parallel_print_image as ppi
import pandas as pd
import os
import numpy as np
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def error_calculate(r_size, ap_size, v_origin, w_origin, ite, wh_seed, c_mode, nmfqp, t_flag):
    theta_start = 5
    seeds = ff.two_seeds(wh_seed)
    v_nmf_error = np.zeros(ite)
    v_snmf_error = np.zeros(ite)
    w_nmf_error = np.zeros(ite)
    w_snmf_error = np.zeros(ite)

    n_size, m_size = v_origin.shape
    w = ff.generate_w(n_size, r_size, seeds[0], c_mode=c_mode)
    h = ff.generate_h(r_size, m_size, seeds[1], c_mode=c_mode)
    v_s = ff.uniform_sampling(v_origin, ap_size, 0)

    # NMF calculate  ---------------------------------------------------------------------------------------------------
    if c_mode == 2:
        theta_w = theta_h = theta_start

    if nmfqp:
        print("NMF matrix H is calculated by QP.")
    for i in range(0, ite):
        if c_mode == 2:
            w, h, theta_w, theta_h = ff.fgd_update(v_origin, w, h, theta_w, theta_h)
        else:
            w, h = ff.update(v_origin, w, h, c_mode)

        if nmfqp:
            h_result = ff.calculate_h(v_origin, w, print_interim=False)
        else:
            h_result = h

    # v evaluate  -------------------
        v_nmf_error[i] = np.linalg.norm(v_origin - np.dot(w, h_result)) ** 2
    # w evaluate  -------------------
        if t_flag:
            d = np.linalg.lstsq(h.T, w_origin)[0]
            w_nmf_error[i] = np.linalg.norm(w_origin - np.dot(h.T, d)) ** 2
        else:
            d = np.linalg.lstsq(w, w_origin)[0]
            w_nmf_error[i] = np.linalg.norm(w_origin - np.dot(w, d)) ** 2

        if (i == 0) | (i % 100 == 99):
                print("NMF ( r=" + str(r_size) + "  k=" + str(ap_size) + " ) : " + str(i + 1) + " times update")

    # else:
    #     for i in range(0, ite):
    #         if c_mode != 2:
    #             w, h = ff.update(v_origin, w, h, c_mode)
    #         else:
    #             w, h, theta_w, theta_h = ff.fgd_update(v_origin, w, h, theta_w, theta_h)
    #
    #         # v evaluate  -------------------
    #         v_nmf_error[i] = np.linalg.norm(v_origin - np.dot(w, h)) ** 2
    #         # w evaluate  -------------------
    #         d = np.linalg.lstsq(w, w_origin)[0]
    #         w_nmf_error[i] = np.linalg.norm(w_origin - np.dot(w, d)) ** 2
    #
    #         if (i == 0) | (i % 2000 == 1999):
    #             print("NMF ( r=" + str(r_size) + "  k=" + str(ap_size) + " ) : " + str(i + 1) + " times update")

    # SNMF calculate  --------------------------------------------------------------------------------------------------
    if c_mode == 2:
        theta_w = theta_h = theta_start

    w_s = ff.generate_w(n_size, r_size, seeds[0], c_mode=c_mode)
    h_s = ff.generate_h(r_size, ap_size, seeds[1], c_mode=c_mode)

    for i in range(0, ite):
        if c_mode == 2:
            w_s, h_s, theta_w, theta_h = ff.fgd_update(v_s, w_s, h_s, theta_w, theta_h)
        else:
            w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)

        # v evaluate  -------------------
        os_M = ff.calculate_h(v_origin, w_s, print_interim=False)
        v_snmf_error[i] = np.linalg.norm(v_origin - np.dot(w_s, os_M)) ** 2
        # w evaluate  -------------------
        if t_flag:
            d_s = np.linalg.lstsq(os_M.T, w_origin)[0]
            w_snmf_error[i] = np.linalg.norm(w_origin - np.dot(os_M.T, d_s)) ** 2
        else:
            d_s = np.linalg.lstsq(w_s, w_origin)[0]
            w_snmf_error[i] = np.linalg.norm(w_origin - np.dot(w_s, d_s)) ** 2

        if (i == 0) | (i % 100 == 99):
            print("SketchingNMF ( r={}  k={} seed={} ) : {} times update".format(r_size, ap_size, wh_seed, i + 1))

    if t_flag:
        return v_nmf_error, v_snmf_error, w_nmf_error, w_snmf_error, h.T, w.T, h_s.T, w_s.T, os_M
    else:
        return v_nmf_error, v_snmf_error, w_nmf_error, w_snmf_error, w, h, w_s, h_s, os_M


tmp = sys.argv

iteration_list = []
# iteration_list = [50]
# for i in range(250, 5001, 250):
#     iteration_list.append(i)
for i in range(100, 1001, 100):
    iteration_list.append(i)

# program_num = "17(for_MIRU)"
program_num = "18(for_MIRU_FGD_bag_fix)"
# program_num = "19(for_MIRU_not_transpose)"

c_mode = 0  # 0:MU NMF  1:HALS NMF  2:FGD  3:GCD

NMFQP = False
Time_Measurement = True
Error_Cal = False

# prepare parameter in python  -----------------------------------------------------------------------------------------
if c_mode == 0:
    c_method = "MU"
elif c_mode == 1:
    c_method = "HALS"
elif c_mode == 2:
    c_method = "FGD"
elif c_mode == 3:
    c_method = "GCD"

# parameter from bash  -------------------------------------------------------------------------------------------------
n = int(tmp[8])
m = int(tmp[9])
r = int(tmp[1])
approximate_size = int(tmp[2])
wh_seed = int(tmp[3])  # base is 1
v_seed = wh_seed - 1

if tmp[5] == "test":
    program_num = "ttest"

    iteration_list = []
    for i in range(50, 501, 50):
        iteration_list.append(i)

progress = (int(tmp[6]) - 1) * len(iteration_list) + 1
all_program_num = int(tmp[7]) * len(iteration_list)

V_origin, W_origin, H_origin = ff.generate_wh(n, m, r, v_seed)
T_flag = False
if V_origin.shape[0] > V_origin.shape[1]:
    V_origin = V_origin.T
    T_flag = True
    print("V is transposed")

for ite_i, iteration in enumerate(iteration_list):
    w_path = "/home/ionishi/mnt/workspace/sketchingNMF/random_matrix/{}/{}_iteration{}/r,k/r={}/k={}" \
        .format(program_num, c_method, iteration, r, approximate_size)
    # make directory
    os.makedirs(w_path + "/time", exist_ok=True)

    # time measurement  ------------------------------------------------------------------------------------------------
    program_code = ff.program_name(0, c_mode, n, m, r, approximate_size, iteration, v_seed, wh_seed, program_num)
    print("start  random_matrix_" + program_code + "  >>>>>\n")
    print([n, m, V_origin.shape])
    if Time_Measurement:

        t_result = tm.parallel_time_measurement(r, approximate_size, V_origin, iteration, wh_seed, c_mode)

        # save time result
        tf_result = pd.DataFrame({"NMF time": t_result[0], "SNMF time": t_result[1]}, index=[0])
        tf_result.to_csv(w_path + "/time/" + program_code + ".csv")

        print("\n{} / {} finish time measurement! {}".format(progress + ite_i, all_program_num, program_code))

    # error calculate  -------------------------------------------------------------------------------------------------
    if (iteration == max(iteration_list)) & Error_Cal:
        error_cal_ = True
    else:
        error_cal_ = False

    if error_cal_:
        v_nmf_error, v_snmf_error, w_nmf_error, w_snmf_error, W, H, W_s, H_s, QP_matrix \
            = error_calculate(r, approximate_size, V_origin, W_origin, iteration, wh_seed, c_mode, NMFQP, T_flag)

        # save result  -------------------------------------------------------------------------------------------------
        os.makedirs(w_path + "/graph", exist_ok=True)
        os.makedirs(w_path + "/error", exist_ok=True)
        os.makedirs(w_path + "/matrix", exist_ok=True)
        # v evaluate result  ---------------------
        program_code = ff.program_name(2, c_mode, n, m, r, approximate_size, iteration, v_seed, wh_seed, program_num)

        # save error list
        e_result = pd.DataFrame([v_nmf_error, v_snmf_error], index=["NMF error", "SNMF error"])
        e_result.T.to_csv(w_path + "/error/" + program_code + "_error.csv")

        # save matrix
        np.savetxt(w_path + "/matrix/w_" + program_code + ".csv", W, delimiter=",")
        np.savetxt(w_path + "/matrix/h_" + program_code + ".csv", H, delimiter=",")
        np.savetxt(w_path + "/matrix/W_s_" + program_code + ".csv", W_s, delimiter=",")
        np.savetxt(w_path + "/matrix/H_s_" + program_code + ".csv", H_s, delimiter=",")
        if T_flag:
            np.savetxt(w_path + "/matrix/W_QP_" + program_code + ".csv", QP_matrix, delimiter=",")
        else:
            np.savetxt(w_path + "/matrix/H_QP_" + program_code + ".csv", QP_matrix, delimiter=",")

        ppi.print_images(c_mode, program_code, program_num, V_origin, W, H, W_s, H_s, r, approximate_size, v_nmf_error,
                         v_snmf_error, iteration, wh_seed, False)

        # w evaluate result  -------------------
        # save error list
        program_code = ff.program_name(3, c_mode, n, m, r, approximate_size, iteration, v_seed, wh_seed, program_num)
        e_result = pd.DataFrame([w_nmf_error, w_snmf_error], index=["NMF error", "SNMF error"])
        e_result.T.to_csv(w_path + "/error/" + program_code + "_error.csv")

        ppi.print_images(c_mode, program_code, program_num, V_origin, W, H, W_s, H_s, r, approximate_size, w_nmf_error,
                         w_snmf_error, iteration, wh_seed, False)
        print("\n{} / {} finish error_calculate  {}  >>>>>\n".format(progress + ite_i, all_program_num, program_code))

