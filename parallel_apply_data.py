import functionfile as ff
import v_error_eval as v_eval
import time_measurement as tm
import image_processing_function as ipf
import parallel_print_image as ppi
import pandas as pd
import os
import numpy as np
import sys
import matplotlib
matplotlib.use('agg')

tmp = sys.argv

program_num = "12(for_MIRU_not_Trans)"

c_mode = 0  # 0:MU NMF  1:HALS NMF  2:FGD  3:GCD

NMFQP = False
Time_Measurement = True
output_TM_matrix = True
Error_Cal = True

# prepare parameter in python  -----------------------------------------------------------------------------------------
if c_mode == 0:
    c_bar_max = 70
    iteration_list = [5000]
    c_method = "MU"
elif c_mode == 1:
    c_bar_max = 70
    iteration_list = [1000]
    c_method = "HALS"
elif c_mode == 2:
    c_bar_max = 70
    if Error_Cal:
        iteration_list = [1000]
    else:
        iteration_list = []
        for i in range(50, 1001, 50):
            iteration_list.append(i)
    c_method = "FGD"
elif c_mode == 3:
    c_bar_max = 70
    if Error_Cal:
        iteration_list = [1000]
    else:
        iteration_list = []
        for i in range(50, 1001, 50):
            iteration_list.append(i)
    c_method = "GCD"

# parameter from bash  -------------------------------------------------------------------------------------------------
if tmp[5] == "test":
    program_num = "test"
    iteration_list = [5]
    test_flag = True
else:
    test_flag = False

if tmp[4] == "cbcl":
    use_data = "CBCL/train"
    image_num_list = [0, 500, 1000, 1500]
elif tmp[4] == "yale":
    use_data = "YaleFD/faces"
    image_num_list = [0, 20, 40, 60]

r = int(tmp[1])
approximate_size = int(tmp[2])
wh_seed = int(tmp[3])  # base is 1
progress = (int(tmp[6]) - 1) * len(iteration_list) + 1
all_program_num = int(tmp[7]) * len(iteration_list)

r_path = "/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + use_data
w_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/r,k/r={}/k={}" \
    .format(use_data, program_num, c_method, r, approximate_size)

V, im_var, im_hol = ipf.read_pgm(r_path)

# when V size is n > m , transpose V
t_flag = False
if V.shape[0] > V.shape[1]:
    V = V.T
    t_flag = True
n, m = V.shape

if (approximate_size == m) | test_flag:
    SNMF_only = False
else:
    SNMF_only = True

for ite_i, iteration in enumerate(iteration_list):
    program_code = "realdata_" + ff.program_name(0, c_mode, n, m, r, approximate_size, iteration, 0, wh_seed, program_num)
    os.makedirs(w_path + "/time", exist_ok=True)
    print("start  " + program_code + "  >>>>>\n")

    # time measurement  ------------------------------------------------------------------------------------------------
    if Time_Measurement:
        if output_TM_matrix:
            t_result, W, H, W_s, H_s, small_matrix = \
                tm.parallel_time_measurement(r, approximate_size, V, iteration, wh_seed, c_mode, snmf_only=SNMF_only, output_matrices=output_TM_matrix, t_flag=t_flag)
            program_code = "realdata_" + ff.program_name(2, c_mode, n, m, r, approximate_size, iteration, 0, wh_seed, program_num)
            os.makedirs(w_path + "/matrix", exist_ok=True)
            if not SNMF_only:
                np.savetxt(w_path + "/matrix/w_" + program_code + ".csv", W, delimiter=",")
                np.savetxt(w_path + "/matrix/h_" + program_code + ".csv", H, delimiter=",")
            np.savetxt(w_path + "/matrix/W_s_" + program_code + ".csv", W_s, delimiter=",")
            np.savetxt(w_path + "/matrix/H_s_" + program_code + ".csv", H_s, delimiter=",")
            if V.shape[0] <= V.shape[1]:
                np.savetxt(w_path + "/matrix/H_small_" + program_code + ".csv", small_matrix, delimiter=",")
            else:
                np.savetxt(w_path + "/matrix/W_small_" + program_code + ".csv", small_matrix, delimiter=",")
            program_code = "realdata_" + ff.program_name(0, c_mode, n, m, r, approximate_size, iteration, 0, wh_seed, program_num)
        else:
            t_result = tm.parallel_time_measurement(r, approximate_size, V, iteration, wh_seed, c_mode, snmf_only=SNMF_only)
            print("\n{} / {} finish time measurement  {}  >>>>>\n".format(progress + ite_i, all_program_num, program_code))

        # save time result
        if SNMF_only:
            tf_result = pd.DataFrame({"SNMF time": t_result[1]}, index=[0])
        else:
            tf_result = pd.DataFrame({"NMF time": t_result[0], "SNMF time": t_result[1]}, index=[0])
        tf_result.to_csv(w_path + "/time/" + program_code + ".csv")

    # calculate --------------------------------------------------------------------------------------------------------
    if Error_Cal:
        os.makedirs(w_path + "/graph", exist_ok=True)
        os.makedirs(w_path + "/error", exist_ok=True)
        os.makedirs(w_path + "/matrix", exist_ok=True)

        program_code = "realdata_" + ff.program_name(2, c_mode, n, m, r, approximate_size, iteration, 0, wh_seed, program_num)
        nmf_error, snmf_error, W, H, W_s, H_s, small_matrix = \
            v_eval.parallel_v_error_eval(r, approximate_size, V, iteration, wh_seed, c_mode, NMFQP, t_flag, snmf_only=SNMF_only)

        print("\n {} / {} finish  calculate !  {}".format(progress + ite_i, all_program_num, program_code))

        # save matrix
        if not SNMF_only:
            np.savetxt(w_path + "/matrix/w_" + program_code + ".csv", W, delimiter=",")
            np.savetxt(w_path + "/matrix/h_" + program_code + ".csv", H, delimiter=",")
        np.savetxt(w_path + "/matrix/W_s_" + program_code + ".csv", W_s, delimiter=",")
        np.savetxt(w_path + "/matrix/H_s_" + program_code + ".csv", H_s, delimiter=",")

        if V.shape[0] <= V.shape[1]:
            np.savetxt(w_path + "/matrix/H_small_" + program_code + ".csv", small_matrix, delimiter=",")
        else:
            np.savetxt(w_path + "/matrix/W_small_" + program_code + ".csv", small_matrix, delimiter=",")

        # save error list
        if not SNMF_only:
            e_result = pd.DataFrame([nmf_error, snmf_error], index=["NMF error", "SNMF error"])
        else:
            e_result = pd.DataFrame([snmf_error], index=["SNMF error"])
        e_result.T.to_csv(w_path + "/error/" + program_code + "_error.csv")

        # plot graph
        # if t_flag:
        #     V = V.T
        # ppi.print_images(c_mode, program_code, program_num, V, W, H, W_s, H_s, r, approximate_size, nmf_error,
        #                  snmf_error, iteration, wh_seed, True, im_var=im_var, im_hol=im_hol, use_data=use_data,
        #                  image_num_list=image_num_list, c_bar_max=c_bar_max)
    print("\n{} / {} finish  {}  >>>>>\n".format(progress + ite_i, all_program_num, program_code))
